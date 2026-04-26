from typing import Any

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import string

ALPHABET = list(string.ascii_uppercase)
N_ALPHABET = len(ALPHABET)

SHEETS_AUTH = "https://www.googleapis.com/auth/spreadsheets"

# SCOPES:
# "": all
# ".readonly"


def load_creds(
    token_fp: str = "./GSHEET_TOKEN.json",
    secrets_fp: str = "./GSHEET_CREDENTIALS.json",
    scopes: list[str] = [""],
):
    # If modifying scopes, delete token_fp.
    scopes = [f"{SHEETS_AUTH}{sc}" for sc in scopes]

    creds = None
    if os.path.exists(token_fp):
        creds = Credentials.from_authorized_user_file(
            token_fp, scopes
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = (
                InstalledAppFlow.from_client_secrets_file(
                    secrets_fp, scopes
                ).run_local_server(port=0)
            )
        with open(token_fp, "w") as token:
            token.write(creds.to_json())

    return creds


CEG_PFOLIO = "1R1SxAabcEu8sa7t4NQqVTILOqU3YT2P758559pjwqOc"


def cell(row: int | None, col: int | None):
    row_s = col_s = ""
    if col is not None:
        col_i_0, col_i_1 = divmod(col, N_ALPHABET)
        try:
            col_s = (
                ALPHABET[col_i_1]
                if col_i_0 == 0
                else ALPHABET[col_i_0 - 1]
                + ALPHABET[col_i_1 - 1]
            )
        except:
            raise ValueError((col, col_i_0, col_i_1))
    if row is not None:
        row_s = str(row + 1)
    return col_s + row_s


def read(
    id: str,
    sheet: str,
    start: str,
    end: str,
    token_fp: str = "./GSHEET_TOKEN.json",
    secrets_fp: str = "./GSHEET_CREDENTIALS.json",
) -> list[list[Any]]:
    creds = load_creds(token_fp, secrets_fp)
    service = build("sheets", "v4", credentials=creds)
    return (
        service.spreadsheets()
        .values()
        .get(
            spreadsheetId=id, range=f"{sheet}!{start}:{end}"
        )
        .execute()
        .get("values", [])
    )


import datetime as dt


def encode(v):
    if isinstance(v, dt.date):
        return v.isoformat()
    elif isinstance(v, dt.datetime):
        return v.isoformat()
    return v


def write(
    id: str,
    sheet: str,
    values: list[list[Any]],
    row: int | None = None,
    col: int | None = None,
    value_input_option: str = "USER_ENTERED",
    token_fp: str = "./GSHEET_TOKEN.json",
    secrets_fp: str = "./GSHEET_CREDENTIALS.json",
):
    row = 0 if row is None else row
    col = 0 if col is None else col

    start = cell(row, col)

    height = len(values)
    width = max([len(r) for r in values])

    end = cell(col=col + width, row=row + height)  # -1?

    creds = load_creds(token_fp, secrets_fp)
    service = build("sheets", "v4", credentials=creds)

    body = {
        "values": [[encode(v) for v in r] for r in values]
    }
    result = (
        service.spreadsheets()
        .values()
        .update(
            spreadsheetId=id,
            range=f"{sheet}!{start}:{end}",
            valueInputOption=value_input_option,
            body=body,
        )
        .execute()
    )

    assert result.get("updatedCells", 0) > 0, (
        result,
        height,
        width,
    )

    return result
