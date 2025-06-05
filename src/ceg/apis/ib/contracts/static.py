
from types import SimpleNamespace
from .prices.db import E, T, C, StringNamespace, Contract

class NestedStringNamespace(SimpleNamespace):

    def __getattribute__(self, k: str) -> StringNamespace:
        return SimpleNamespace.__getattribute__(self, k)

# TODO: just use classes

CONTRACTS: dict[tuple[str, str], Contract] = {
    (T.GENERIC, "ES"): Contract.new(
        T.GENERIC,
        "ES",
        exchange=E.CME,
        currency=C.USD,
    )
}

INDICES = StringNamespace(
    ITRX_HY_5Y_TR="ITRXTX5I",
    ITRX_IG_5Y_TR="ITRXTE5I",

    ITRX_HY_5Y_VOL="VIXXO",
    ITRX_IG_5Y_VOL="VIXIE",

    CDX_HY_5Y_1M_VOL="VIXHY",
    CDX_IG_5Y_1M_VOL="VIXIG",
)

FUT = NestedStringNamespace(
    # CME except djia=cbot
    CRUDE=StringNamespace(
        BRENT="BZ", # nymex
        BRENT_ICE="COIL", # ipe
        LIGHT_SWEET="CL", # nymex
        LIGHT_SWEET_MINI="QM",
        WTI="WTI", # ipe,
        WTI_MICRO="MCL", # nymex
    ),
    US_EQ_MINI=StringNamespace(
        NQ_BIO="BQX",
        DJIA="YM",
        NQ_100="NQ",
        NQ_COMP="QCN",
        PHLX_SEMIS="SPSOX",
        R1K="RS1",
        R1K_GROWTH="RSG",
        R1K_VALUE="RSV",
        R2K="RTY",
        SP_500="ES",
        SP_500_ESG="SPXESUP",
        SP_MID_400="EMD",
        SP_SMALL_600="SMC",
        VIX="VXM",
        # various micros
    ),
    EU_EQ_MINI=StringNamespace(
        IBEX_35="IBEX",
    ),
    AP_EQ_MINI=StringNamespace(
        TOPIX="MNTPX", # ose.jpn
    ),
    US_GOV = StringNamespace(
        # cbot
        Y30Y_MICRO="30Y",
        P20Y="TWE",
        P10Y_ULTRA_MICRO="MTN",
        P10Y_ULTRA="TN",
        P10Y="ZN",
        Y10Y_MICRO="10Y", # micro
        Y05Y_MICRO="5YY",
        P05Y="ZF",
        P03Y="Z3N",
        P02Y="ZT",
        Y02Y_MICRO="2YY",
        P13W="TBF3",
        T_BOND_ULTRA_MICRO="MWN",
        T_BOND="ZB",
        T_BOND_ULTRA="UB",
    ),
    AU_GOV=StringNamespace(
        P10Y="XT", # 6%
        P03Y="YT",
    ),
    KO_GOV=StringNamespace(
        P30Y="30KTB",
        P10Y="XLKTB",
        P03Y="3KTB",
    )
)

ISHARES = StringNamespace(
    DEFENSE="ITA",
    MATERIALS="IYM",
    BROKERS="IEDI",
    ENERGY="IYE",
    FIN_SERV="IYG",
    FINANCIALS="IYF",
    HEALTHCARE="IYH",
    HEALH_PROVD="IHF",
    HOME_CONSTR="ITB",
    INDUSTRY="IYJ",
    INSURANCE="IAK",
    MED_DEV="IHI",
    DRILLING="IEO",
    DRILLING_SERV="IEZ",
)
# TODO: various bonds, yield, divs, EM, growth, small, mid cap
THEMES = StringNamespace(
    CHINA="GXC",
    DIVIDEND="SDY",
)
SP_SECTORS = StringNamespace(
    DEFENSE="XAR",
    BANKS="KBE",
    BIOTECH="XBI",
    CAP_MARKETS="KCE",
    HEALTH_EQUIP="XHE",
    HEALTH_SERV="XHS",
    HOMEBUILD="XHB",
    INSURACE="KIE",
    MINING="XME",
    DRILLING_SERV="XES",
    DRILLING="XOP",
    PHARMA="XPH",
    BANKING_REGIONAL="KRE",
    RETAIL="XRT",
    SEMIS="XSD",
    SOFTWARE="XSW",
    TELCO="XTL",
    TRANSPORT="XTN",
)
SPDR_SECTORS = StringNamespace(
    COMMUN_SERV="XLC",
    ENERGY="XLE",
    FINANCIALS="XLF",
    MATERIALS="XLB",
    UTILITIES="XLU",
    HEALTHCARE="XLV",
    CONS_STAPLES="XLP",
    INDUSTRY="XLI",
    REAL_ESTATE="XLRE",
    TECHNOLOGY="XLK",
)
SX5E = "FEZ"
MSCI_ACWI_EX_US = "CWI"

BONDS = [
    "10Y US GOV YIELD, TNX index on CBOE",
    "5Y US GOV INDEX, FVX index on CBOE",
    "30Y US GOV INDEX, TYX index on CBOE"
]

OTHERS = [
    dict(
        symbol="IBXXIBHY",
        name="IBOXX ISHARES HY CORP INDEX TR CBOE",
    ),
    dict(
        symbol="IBSSIBIG",
        name="IBOXX ISHARES IG CORP INDEX TR CBOE",
    ),
    dict(
        symbol="CVK",
        name="ISHARES SP SMALLCAP 600/BARRA VALUE PSE"
    ),
    dict(
        symbol="IJR",
        name="ISHARES CORE SP SMALLCAP E ARCA"
    ),
    dict(
        symbol="XIWM",
        name="ISHARES RUSSELL 2000 INDEX FUND AMEX",
    ),
    dict(
        symbol="IWM",
        name="ISHARES RUSSELL 2000 ETF ARCA",
    ),
    dict(
        symbol="IVV",
        name="ISHARES CORE SP 500 ETF ARCA",
    ),
    dict(
        symbol="IVW",
        name="SHARES CORE SP 500 GROWTH ETF ARCA",
    ),
    dict(
        symbol="ITOT",
        name="SHARES CORE SP 500 TOTAL US ARCA",
    ),
    dict(
        symbol="IWF",
        name="ISHARES RUSSELL 1000 GROWTH ARCA"
    ),
    dict(
        symbol="IEFA",
        name="ISHARES CORE MSCI EAFA ETF BATS",
    ),
    dict(
        symbol="IEMG",
        name="ISHARES CORE MSCI EMERGING ARCA",
    ),
    dict(
        symbol="TLT",
        name="ISHARES 20+ Y TREASURY BD NASDAQ",
    ),
    dict(
        symbol=""
    )
]



#  ------------------
