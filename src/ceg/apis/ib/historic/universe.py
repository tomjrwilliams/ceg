from ..db import E, T, C, StringNamespace

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
