"""
src/features/identity_mismatch.py
==================================
Mismatch and consistency features derived from identity and address signals.

In e-commerce fraud, a common adversarial pattern is a discrepancy between
the purchaser's identity and the recipient's:
  - A stolen card billed to New York being shipped to a freight forwarder
    with a ``R_emaildomain`` the cardholder has never used.
  - A ``P_emaildomain`` of gmail.com paired with a ``R_emaildomain`` of a
    temporary/disposable mail service.
  - A card (``card1``) that has historically always shipped to ``addr1=100``
    suddenly appearing with ``addr1=999`` — a geographic anomaly.

Features produced
-----------------
1. **Email mismatch**
   - ``email_match``             : 1 if P_ and R_ domains are identical
   - ``p_email_tld``             : TLD of purchaser domain (gmail, yahoo …)
   - ``r_email_tld``             : TLD of recipient domain
   - ``email_tld_match``         : 1 if both TLDs are the same (looser than exact)
   - ``p_email_is_free``         : 1 if P domain is a known free-mail provider
   - ``r_email_is_free``         : 1 if R domain is a known free-mail provider
   - ``both_free_email``         : 1 if both are free — not inherently fraudulent,
                                   but combined with other signals it matters

2. **Address / geographic consistency** (per card1 entity)
   - ``card1_addr1_is_mode``     : 1 if current addr1 matches the most common
                                   addr1 seen for this card in the training set
   - ``card1_addr2_is_mode``     : same for addr2 (country-level)
   - ``card1_n_unique_addr1``    : number of distinct billing regions seen for
                                   this card (high = suspicious)
   - ``card1_n_unique_addr2``    : same for addr2

3. **Identity presence**
   - ``has_identity``            : 1 if ANY identity column is non-null
   - ``identity_completeness``   : fraction of identity columns that are non-null
                                   (0 = no enrichment; 1 = fully enriched)

Leakage handling
----------------
Address consistency features (mode addr per card) are computed from the
**training set** via ``fit_address_encoders(train_df)`` and then applied to
any split.  Call ``fit_address_encoders`` once on train; pass the result to
``compute_identity_features`` for val/test.

``pipeline.build_features()`` handles this automatically.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common free / consumer email providers (non-exhaustive but covers >90% of
# typical e-commerce traffic).  Disposable / temp-mail domains are a
# separate, evolving list — flag them in a dedicated feature if available.
_FREE_EMAIL_PROVIDERS: FrozenSet[str] = frozenset({
    "gmail", "yahoo", "hotmail", "outlook", "aol", "icloud",
    "live", "msn", "protonmail", "ymail", "mail", "gmx",
    "zoho", "inbox", "yandex", "qq", "163", "126",
})

# Identity columns (all id_XX + device columns)
# We detect them dynamically but keep a prefix list
_IDENTITY_PREFIXES: tuple[str, ...] = ("id_", "DeviceType", "DeviceInfo")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_tld(domain_series: pd.Series) -> pd.Series:
    """
    Extract the top-level domain name from an email domain string.

    e.g. "gmail.com" → "gmail", "yahoo.co.uk" → "yahoo", NaN → "unknown"

    Parameters
    ----------
    domain_series : pd.Series
        Raw email domain column (e.g. P_emaildomain).

    Returns
    -------
    pd.Series
        Lower-cased TLD strings with NaN filled as "unknown".
    """
    return (
        domain_series
        .astype(str)
        .str.lower()
        .str.split(".")
        .str[0]
        .replace("nan", "unknown")
        .fillna("unknown")
    )


def _is_free_email(tld_series: pd.Series) -> pd.Series:
    """Return int8 series: 1 if TLD is a known free-mail provider."""
    return tld_series.isin(_FREE_EMAIL_PROVIDERS).astype(np.int8)


def _get_identity_cols(df: pd.DataFrame) -> list[str]:
    """Detect identity columns present in the DataFrame."""
    return [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in _IDENTITY_PREFIXES)
    ]


# ---------------------------------------------------------------------------
# 1. Email mismatch features
# ---------------------------------------------------------------------------

def compute_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive email-domain mismatch and type features.

    Works gracefully when either ``P_emaildomain`` or ``R_emaildomain``
    is missing (NaN) — mismatch flags become 0 and TLD becomes "unknown".

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        ``df`` with email feature columns appended.
    """
    out = df.copy()

    p_col = "P_emaildomain"
    r_col = "R_emaildomain"

    # Exact domain match (NaN == NaN is True in pandas .eq, so handle explicitly)
    p_raw = df[p_col].astype(str).str.lower().replace("nan", np.nan) if p_col in df.columns else pd.Series("unknown", index=df.index)
    r_raw = df[r_col].astype(str).str.lower().replace("nan", np.nan) if r_col in df.columns else pd.Series("unknown", index=df.index)

    # 1 if both non-null and identical; 0 otherwise (including NaN cases)
    out["email_match"] = (
        p_raw.notna() & r_raw.notna() & (p_raw == r_raw)
    ).astype(np.int8)

    # TLD extraction
    p_tld = _extract_tld(p_raw if p_col in df.columns else pd.Series(np.nan, index=df.index))
    r_tld = _extract_tld(r_raw if r_col in df.columns else pd.Series(np.nan, index=df.index))

    out["p_email_tld"]   = p_tld
    out["r_email_tld"]   = r_tld
    out["email_tld_match"] = (p_tld == r_tld).astype(np.int8)

    # Free-email flags
    out["p_email_is_free"] = _is_free_email(p_tld)
    out["r_email_is_free"] = _is_free_email(r_tld)
    out["both_free_email"] = (
        (out["p_email_is_free"] == 1) & (out["r_email_is_free"] == 1)
    ).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# 2. Address consistency features (requires train-time fitting)
# ---------------------------------------------------------------------------

def fit_address_encoders(
    train_df: pd.DataFrame,
    entity_col: str = "card1",
    addr_cols: list[str] = ("addr1", "addr2"),
) -> Dict[str, object]:
    """
    Build per-card address consistency lookups from the training set.

    For each (entity, address_col) pair we store:
    - ``mode``          : most common address value seen for this card
    - ``n_unique``      : number of distinct address values for this card

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition only.
    entity_col : str
        Grouping entity (default "card1").
    addr_cols : tuple of str
        Address columns to analyse (default ("addr1", "addr2")).

    Returns
    -------
    dict
        Encoders dict keyed by ``"{entity_col}_{addr_col}_mode"`` and
        ``"{entity_col}_{addr_col}_n_unique"``.
    """
    encoders: Dict[str, object] = {}

    if entity_col not in train_df.columns:
        logger.warning("address encoder: '%s' not found — skipping.", entity_col)
        return encoders

    for addr_col in addr_cols:
        if addr_col not in train_df.columns:
            logger.warning("address encoder: '%s' not found — skipping.", addr_col)
            continue

        grp = train_df.groupby(entity_col, observed=True)[addr_col]

        # Mode: most frequent value per card (pd.Series.mode()[0] can error on
        # empty groups, so we use a safe agg)
        mode_series = grp.agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        n_unique_series = grp.nunique()

        mode_key    = f"{entity_col}_{addr_col}_mode"
        nunique_key = f"{entity_col}_{addr_col}_n_unique"
        encoders[mode_key]    = mode_series
        encoders[nunique_key] = n_unique_series

        logger.info(
            "Address encoder fitted: %s  |  %d unique %s values.",
            mode_key, len(mode_series), entity_col,
        )

    return encoders


def compute_address_features(
    df: pd.DataFrame,
    encoders: Dict[str, object],
    entity_col: str = "card1",
    addr_cols: list[str] = ("addr1", "addr2"),
) -> pd.DataFrame:
    """
    Compute address consistency features using pre-fitted ``encoders``.

    Parameters
    ----------
    df : pd.DataFrame
    encoders : dict
        Output of ``fit_address_encoders(train_df)``.
    entity_col : str
    addr_cols : tuple of str

    Returns
    -------
    pd.DataFrame
        ``df`` with address consistency columns appended.
    """
    out = df.copy()

    if entity_col not in df.columns:
        logger.warning("address features: '%s' not found — skipping.", entity_col)
        return out

    for addr_col in addr_cols:
        mode_key    = f"{entity_col}_{addr_col}_mode"
        nunique_key = f"{entity_col}_{addr_col}_n_unique"

        if mode_key not in encoders:
            logger.warning(
                "address features: encoder '%s' not found — skipping.", mode_key
            )
            continue

        mode_map    = encoders[mode_key]
        nunique_map = encoders[nunique_key]

        # Map current card → historical mode address
        card_mode    = df[entity_col].map(mode_map)
        card_n_unique = df[entity_col].map(nunique_map).fillna(1).astype(np.int16)

        # is_mode: does current addr match historical mode?
        #   NaN current addr → 0 (unknown address is suspicious)
        #   Unseen card      → 0 (no history = cannot confirm)
        if addr_col in df.columns:
            current_addr = df[addr_col]
            is_mode = (
                current_addr.notna() &
                card_mode.notna() &
                (current_addr == card_mode)
            ).astype(np.int8)
        else:
            is_mode = pd.Series(0, index=df.index, dtype=np.int8)

        is_mode_col  = f"{entity_col}_{addr_col}_is_mode"
        n_unique_col = f"{entity_col}_{addr_col}_n_unique"
        out[is_mode_col]  = is_mode
        out[n_unique_col] = card_n_unique

    return out


# ---------------------------------------------------------------------------
# 3. Identity presence / completeness
# ---------------------------------------------------------------------------

def compute_identity_presence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute identity presence and completeness features.

    ``has_identity`` is a simple binary flag.
    ``identity_completeness`` is a continuous [0,1] score indicating what
    fraction of identity columns are populated.  A fully enriched transaction
    (all 41 identity columns present) gets a score of 1.0.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    id_cols = _get_identity_cols(df)
    out = df.copy()

    if not id_cols:
        logger.warning("identity_presence: no identity columns found.")
        out["has_identity"]           = np.int8(0)
        out["identity_completeness"]  = np.float32(0.0)
        return out

    # has_identity: at least one id column is non-null
    id_matrix   = df[id_cols].notna()
    out["has_identity"] = id_matrix.any(axis=1).astype(np.int8)

    # completeness: mean non-null rate across all id columns
    out["identity_completeness"] = id_matrix.mean(axis=1).astype(np.float32)

    logger.debug(
        "Identity presence: %d id columns scanned.  "
        "has_identity rate: %.1f%%",
        len(id_cols),
        out["has_identity"].mean() * 100,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Unified entry point
# ---------------------------------------------------------------------------

def compute_identity_features(
    df: pd.DataFrame,
    address_encoders: Optional[Dict[str, object]] = None,
    entity_col: str = "card1",
    addr_cols: list[str] = ("addr1", "addr2"),
) -> pd.DataFrame:
    """
    Compute all identity and mismatch features in one call.

    Parameters
    ----------
    df : pd.DataFrame
    address_encoders : dict, optional
        Output of ``fit_address_encoders(train_df)``.
        If None, address consistency features are derived from ``df``
        itself — correct only when ``df`` is the training set.
    entity_col : str
        Card entity column (default "card1").
    addr_cols : tuple of str
        Address columns to assess consistency for.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    logger.info("Identity: computing email mismatch features …")
    out = compute_email_features(df)

    if address_encoders is None:
        logger.info(
            "Identity: no address encoders provided — fitting on current df "
            "(only correct if this is the training set)."
        )
        address_encoders = fit_address_encoders(
            out, entity_col=entity_col, addr_cols=list(addr_cols)
        )

    logger.info("Identity: computing address consistency features …")
    out = compute_address_features(
        out, encoders=address_encoders,
        entity_col=entity_col, addr_cols=list(addr_cols),
    )

    logger.info("Identity: computing presence / completeness features …")
    out = compute_identity_presence_features(out)

    new_cols = [c for c in out.columns if c not in df.columns]
    logger.info("Identity: added %d features.", len(new_cols))
    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from src.ingestion.loader import load_and_merge
    from src.ingestion.splitter import time_aware_split

    df    = load_and_merge()
    train, val, _ = time_aware_split(df)

    encoders = fit_address_encoders(train)
    result   = compute_identity_features(train, address_encoders=encoders)

    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\nAdded {len(new_cols)} identity/mismatch features:")
    print(result[["TransactionID", "isFraud"] + new_cols].head(10).to_string())
