import time

import pandas as pd

import requests
import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

OMEKA_API_URL = "https://communitymappingarchive.org/api"
OMEKA_KEY_IDENTITY = os.getenv("OMEKA_KEY_IDENTITY")
OMEKA_KEY_CREDENTIAL = os.getenv("OMEKA_KEY_CREDENTIAL")

DLG_APPRAISAL_DOCS = {
    "https://dlg.usg.edu/record/guan_1633_055-002": "Parcel Appraisal Reports - Block 1 [includes photographs and sketch of properties], 1962-1968",
    "https://dlg.usg.edu/record/guan_1633_055-003": "Parcel Appraisal Reports - Block 2 [includes photographs and sketches of properties], 1962-1966",
    "https://dlg.usg.edu/record/guan_1633_055-004": "Parcel Appraisal Reports - Block 3 [includes photographs and sketches of properties], 1962-1967",
    "https://dlg.usg.edu/record/guan_1633_055-005": "Parcel Appraisal Reports - Block 4 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_055-006": "Parcel Appraisal Reports - Block 5 [includes photographs and sketches of properties], 1962-1968",
    "https://dlg.usg.edu/record/guan_1633_055-007": "Parcel Appraisal Reports - Block 6 [includes photographs and sketches of properties], 1962-1967",
    "https://dlg.usg.edu/record/guan_1633_056-001": "Parcel Appraisal Reports - Block 7 [including photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-002": "Parcel Appraisal Reports - Block 8 [includes photographs and sketches of properties], 1962-1965",
    "https://dlg.usg.edu/record/guan_1633_056-003": "Parcel Appraisal Reports - Block 9 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-004": "Parcel Appraisal Reports - Block 10 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-005": "Parcel Appraisal Reports - Block 11 [includes photographs and sketches of properties], 1962-1965",
    "https://dlg.usg.edu/record/guan_1633_056-006": "Parcel Appraisal Reports - Block 12 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-007": "Parcel Appraisal Reports - Block 13 & 14 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-008": "Parcel Appraisal Reports - Block 15 & 16 [includes photographs and sketches of properties], 1962-1968",
    "https://dlg.usg.edu/record/guan_1633_056-009": "Parcel Appraisal Reports - Block 17 & 18 [includes photographs and sketches of properties], 1962-1966",
    "https://dlg.usg.edu/record/guan_1633_056-010": "Parcel Appraisal Reports - Block 19 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_056-011": "Parcel Appraisal Reports - Block 20 & 21 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_057-001": "Parcel Appraisal Reports - Block 22 [includes photographs and sketches of properties], 1962, 1965",
    "https://dlg.usg.edu/record/guan_1633_057-002": "Parcel Appraisal Reports - Block 23-25 [includes photographs and sketches of properties], 1962-1965",
    "https://dlg.usg.edu/record/guan_1633_057-003": "Parcel Appraisal Reports - Block 26-30 [includes photographs and sketches of properties], 1962-1965",
    "https://dlg.usg.edu/record/guan_1633_057-004": "Parcel reuse appraisals, 1962",
    "https://dlg.usg.edu/record/guan_1633_057-005": "Parcel reuse appraisals, 1964",
    "https://dlg.usg.edu/record/guan_1633_057-006": "Parcel reuse appraisals, 1965",
    "https://dlg.usg.edu/record/guan_1633_057-007": "Parcel reuse appraisals, 1966",
    "https://dlg.usg.edu/record/guan_1633_057-008": "Parcel reuse appraisals, 1968",
    "https://dlg.usg.edu/record/guan_1633_057-009": "Parcel reuse appraisals, 1969",
    "https://dlg.usg.edu/record/guan_1633_058-001": "Estimate market value appraisals, 1964, 1967",
    "https://dlg.usg.edu/record/guan_1633_058-002": "Appraisals Blocks 1-3 [includes photographs and sketches of properties], 1964-1968",
    "https://dlg.usg.edu/record/guan_1633_058-003": "Appraisals Blocks 4-6 [includes photographs and sketches of properties], 1964-1967",
    "https://dlg.usg.edu/record/guan_1633_059-001": "Appraisals Blocks 7-10 [includes photographs and sketches of properties], 1964-1965",
    "https://dlg.usg.edu/record/guan_1633_059-002": "Appraisals Blocks 11-16 [includes photographs and sketches of properties], 1964-1965, 1968",
    "https://dlg.usg.edu/record/guan_1633_059-003": "Appraisals Blocks 17-23 [includes photographs and sketches of properties], 1964-1965, 1968",
    "https://dlg.usg.edu/record/guan_1633_059-004": "Appraisals Blocks 24, 25, 27, 28, 30 [includes photographs and sketches of properties], 1964-1965",
}


def get_items_list():
    Bradberry_1962_photos = pd.read_csv("data/Bradberry appraisals 1962/processed_photos.csv")
    Bradberry_1962_photos["source_dataset"] = "Bradberry appraisals 1962"
    Bradberry_1962_photos["dcterms:date"] = 1962
    # add prefix to path
    Bradberry_1962_photos["path"] = "https://raw.githubusercontent.com/communitymaplab/athens-r51-property-appraisal-photos/refs/heads/main/pipeline/data/Bradberry%20appraisals%201962/processed/" + Bradberry_1962_photos["path"]
    # Remove B5_P15__photo1.jpg (see README)
    Bradberry_1962_photos = Bradberry_1962_photos[
        ~Bradberry_1962_photos["path"].str.contains("B5_P15__photo1.jpg", na=False)
    ]

    # Replace B8__67_jpg__photo1.jpg with B8_P12__photo4.jpg
    Bradberry_1962_photos.loc[Bradberry_1962_photos["path"].str.contains("B8__67_jpg__photo1.jpg", na=False), "parcel"] = "12"
    Bradberry_1962_photos["path"] = Bradberry_1962_photos["path"].str.replace("B8__67_jpg__photo1.jpg", "B8_P12__photo4.jpg")

    Diaz_1964_photos = pd.read_csv("data/Diaz appraisals 1964/processed_photos.csv")
    Diaz_1964_photos["source_dataset"] = "Diaz appraisals 1964"
    Diaz_1964_photos["dcterms:date"] = 1964
    # add prefix to path
    Diaz_1964_photos["path"] = "https://raw.githubusercontent.com/communitymaplab/athens-r51-property-appraisal-photos/refs/heads/main/pipeline/data/Diaz%20appraisals%201964/processed/" + Diaz_1964_photos["path"]
    # Delete B12_P14__photo3.jpg (see README)
    Diaz_1964_photos = Diaz_1964_photos[
        ~Diaz_1964_photos["path"].str.contains("B12_P14__photo3.jpg", na=False)
    ]


    _DIAZ_BLOCK_RANGES_TO_RECORD = (
        ((4, 6), "guan_1633_058-003"),
        ((7, 10), "guan_1633_059-001"),
        ((11, 16), "guan_1633_059-002"),
        ((17, 23), "guan_1633_059-003"),
        ((24, 25), "guan_1633_059-004"),
        ((27, 28), "guan_1633_059-004"),
        ((30, 30), "guan_1633_059-004"),
    )

    _DIAZ_BLOCK_TO_RECORD = {
        "1": "guan_1633_058-002",
        "2": "guan_1633_058-002",
        "2A": "guan_1633_058-002",
        "3": "guan_1633_058-002",
    }

    _BRADBERRY_BLOCK_TO_RECORD = {
        "1": "guan_1633_055-002",
        "2": "guan_1633_055-003",
        "2A": "guan_1633_055-003",
        "3": "guan_1633_055-004",
        "4": "guan_1633_055-005",
        "5": "guan_1633_055-006",
        "6": "guan_1633_055-007",
        "7": "guan_1633_056-001",
        "8": "guan_1633_056-002",
        "9": "guan_1633_056-003",
        "10": "guan_1633_056-004",
        "11": "guan_1633_056-005",
        "12": "guan_1633_056-006",
        "13": "guan_1633_056-007",
        "14": "guan_1633_056-007",
        "15": "guan_1633_056-008",
        "16": "guan_1633_056-008",
        "17": "guan_1633_056-009",
        "18": "guan_1633_056-009",
        "19": "guan_1633_056-010",
        "20": "guan_1633_056-011",
        "21": "guan_1633_056-011",
        "22": "guan_1633_057-001",
        "23": "guan_1633_057-002",
        "24": "guan_1633_057-002",
        "25": "guan_1633_057-002",
        "26": "guan_1633_057-003",
        "27": "guan_1633_057-003",
        "28": "guan_1633_057-003",
        "30": "guan_1633_057-003",
    }


    def _normalize_block_token(block_value: object) -> str:
        block_s = str(block_value).strip().upper()
        if block_s.startswith("B"):
            block_s = block_s[1:]
        return block_s


    def _record_for_block(
        block_value: object,
        block_mapping: tuple[tuple[tuple[int, int], str], ...],
    ) -> str | None:
        block_s = _normalize_block_token(block_value)
        if not block_s:
            return None
        try:
            block_n = int(block_s)
        except ValueError:
            return None
        for (start, end), record in block_mapping:
            if start <= block_n <= end:
                return record
        return None


    def _record_for_block_token(
        block_value: object,
        block_mapping: dict[str, str],
    ) -> str | None:
        block_s = _normalize_block_token(block_value)
        if not block_s:
            return None
        return block_mapping.get(block_s)


    def _source_url_for_row(row: pd.Series) -> str:
        dataset = str(row.get("source_dataset", "")).strip()
        page = str(row.get("source_page_number", "")).strip()
        if not page:
            return ""

        if dataset == "Diaz appraisals 1964":
            record = _record_for_block_token(row.get("block"), _DIAZ_BLOCK_TO_RECORD)
            if record is None:
                record = _record_for_block(row.get("block"), _DIAZ_BLOCK_RANGES_TO_RECORD)
        elif dataset == "Bradberry appraisals 1962":
            record = _record_for_block_token(row.get("block"), _BRADBERRY_BLOCK_TO_RECORD)
        else:
            record = None

        if not record:
            return ""
        return (
            f"https://dlg.usg.edu/record/{record}"
            f"?canvas={str(int(page)-1)}&x=400&y=400&w=1955"
        )


    merged_photos = pd.concat([Bradberry_1962_photos, Diaz_1964_photos])

    merged_photos["dcterms:source"] = merged_photos.apply(_source_url_for_row, axis=1)

    merged_photos = merged_photos.assign(parcel=merged_photos['parcel'].str.split(',')).explode('parcel')

    merged_photos["dcterms:title"] = (
        "Appraisal photograph of Block "
        + merged_photos["block"]
        + " Parcel "
        + merged_photos["parcel"]
    )
    merged_photos['block'] = 'B' + merged_photos['block'].astype(str)
    merged_photos['parcel'] = 'P' + merged_photos['parcel'].astype(str)
    merged_photos["dcterms:title"] += " (" + merged_photos["block"] + merged_photos["parcel"] + ")"

    merged_photos["dcterms:format"] = "JPEG Image"
    merged_photos["dcterms:rights"] = "Public Domain"
    merged_photos["dcterms:isPartOf"] = (
        "Athens, Georgia city records, ms1633, Hargrett Rare Book and Manuscript Library, "
        "The University of Georgia Libraries"
    )
    merged_photos["dcterms:subject"] = "Athens, Georgia; Urban Renewal; R-51; Property Appraisals"
    merged_photos["dcterms:spatial"] = "Athens, Georgia"
    merged_photos[["path", "block", "parcel"]].to_csv("data/merged_photos.csv", index=False)
    merged_photos = merged_photos[
        [
            "path",
            "dcterms:date",
            "dcterms:source",
            "dcterms:rights",
            "dcterms:isPartOf",
            "dcterms:subject",
            "dcterms:spatial",
            "dcterms:title",
            "dcterms:format",
            "source_page_number"
        ]
    ].rename(columns={"path": "bibo:uri"})
    #.to_csv(
    #    "data/omeka_import.csv", index=False
    #)

    return merged_photos
    # bibo:uri - image link
    # Source - DLG LInk
    # dcterms:isPartOf  - Text citation (ms1633)

    '''
    If you have multiple inputs for a single property, you can separate them with a secondary multivalue separator. For example, a work with multiple authors (E.B. White and William Strunk Jr.) with the column for Creator containing "E.B. White;William Strunk Jr" has a semicolon (;) as the multivalue separator. When imported into Omeka S, each of these would appear as a separate entry in the property (Creator: "E.B. White" and Creator: "William Strunk Jr."). Note that the import will be the same whether you leave a space after your separator (as in "E.B. White; William Strunk Jr") or not.
    '''

    '''
    Import as URI reference. You can set the label for a URI by including the desired text after a space, for example: http://example.com Label Text Goes Her
    '''

    # URI label - Use title from dlg, comma, page# XXX

# We will use the Omeka API to create items from the omeka_import.csv file
def upload_to_omeka(items_list):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    # Get field name and numbers from Omeka API
    # Set requests user-agent

    omeka_fields = session.get(
        f"{OMEKA_API_URL}/properties"
    ).json()

    # Get list of current items in Omeka - just store titles
    omeka_current_item_titles = set()
    page = 1
    response = session.get(f"{OMEKA_API_URL}/items?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}")
    number_items = int(response.headers.get("omeka-s-total-results"))
    response = response.json()
    omeka_current_item_titles = set([item["o:title"] for item in response])
    page += 1
    while len(omeka_current_item_titles) < number_items:
        response = session.get(f"{OMEKA_API_URL}/items?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}&page={page}")
        response = response.json()
        omeka_current_item_titles.update([item["o:title"] for item in response])
        page += 1


    #items_list = items_list.head(2).copy()
    items_list = items_list.copy()
    # Append photo number to dcterms:title so that each photo has a unique title - extract from bibo:uri
    photo_id = items_list["bibo:uri"].str.extract(r"__([^.]+)\.jpg", expand=False)
    items_list["dcterms:title"] = (
        items_list["dcterms:title"]
        + " -- "
        + items_list["dcterms:date"].astype(str)
        + "_"
        + photo_id
    )
    
    # Build a term->id map once instead of scanning omeka_fields repeatedly
    term_to_id = {field["o:term"]: field["o:id"] for field in omeka_fields}

    total_rows = len(items_list)
    for row_num, (_, item) in enumerate(items_list.iterrows(), start=1):
        if item["dcterms:title"] in omeka_current_item_titles:
            print(f"Item {item['dcterms:title']} already exists in Omeka")
            continue

        json_body = {}
        for column in items_list.columns:
            if column not in term_to_id and column not in ["bibo:uri"]:
                continue

            raw_value = item[column]
            if pd.isna(raw_value):
                continue

            value_text = str(raw_value).strip()
            if not value_text:
                continue

            if column == "dcterms:source":
                json_body[column] = [
                    {
                        "property_id": term_to_id[column],
                        "type": "uri",
                        "is_public": "1",
                        "@annotation": None,
                        "o:lang": "",
                        "@id": value_text,
                        "o:label": DLG_APPRAISAL_DOCS.get(
                            value_text.split("?", 1)[0].strip(), ""
                        ) + "; Page " + str(item["source_page_number"]),
                    }
                ]
                continue

            if column == "bibo:uri":
                json_body[column] = [
                    {
                        "property_id": 121,
                        "type": "uri",
                        "is_public": "1",
                        "@annotation": None,
                        "o:lang": "",
                        "@id": value_text,
                        "o:label": "",
                    }
                ]
                continue

            values = (
                [v.strip() for v in value_text.split(";") if v.strip()]
                if column == "dcterms:subject"
                else [value_text]
            )

            json_body[column] = [
                {
                    "property_id": term_to_id[column],
                    "type": "literal",
                    "is_public": "1",
                    "@language": "",
                    "@value": value,
                }
                for value in values
            ]

        # Add citation
        json_body["dcterms:bibliographicCitation"] = [
            {
                "property_id": 48,
                "type": "uri",
                "is_public": "1",
                "@annotation": None,
                "o:lang": "",
                "@id": "https://liamengland.com/ms1633-page-old-scl-layout",
                "o:label": "Athens, Georgia city records, ms1633, Hargrett Rare Book and Manuscript Library, The University of Georgia Libraries"
            }
        ]
        post = session.post(
            f"{OMEKA_API_URL}/items?key_identity={OMEKA_KEY_IDENTITY}&key_credential={OMEKA_KEY_CREDENTIAL}",
            headers={"Content-Type": "application/json"},
            json=json_body
        )
        post.raise_for_status()
        print(
            f"Omeka items POST {row_num}/{total_rows} "
            f"(HTTP {post.status_code})"
        )
        if row_num < total_rows:
            time.sleep(1)
        # print(json_body)
    #print(post.text)


if __name__ == "__main__":
    items_list = get_items_list()
    upload_to_omeka(items_list)


# {"dcterms:title":[{"property_id":1,"type":"literal","is_public":"1","@language":"","@value":"Appraisal photograph of Block B1 Parcel P29"}],"dcterms:date":[{"property_id":7,"type":"literal","is_public":"1","@language":"","@value":"1962"}],"dcterms:source":[{"property_id":11,"type":"literal","is_public":"1","@language":"","@value":"https://dlg.usg.edu/record/guan_1633_055-002?canvas=2&x=400&y=400&w=1955"}],"dcterms:rights":[{"property_id":15,"type":"literal","is_public":"1","@language":"","@value":"Public Domain"}],"dcterms:isPartOf":[{"property_id":33,"type":"literal","is_public":"1","@language":"","@value":"Athens, Georgia city records, ms1633, Hargrett Rare Book and Manuscript Library, The University of Georgia Libraries"}],"dcterms:subject":[{"property_id":3,"type":"literal","is_public":"1","@language":"","@value":"Athens, Georgia"},{"property_id":3,"type":"literal","is_public":"1","@language":"","@value":"Urban Renewal"}],"dcterms:spatial":[{"property_id":40,"type":"literal","is_public":"1","@language":"","@value":"Athens, Georgia"}],"dcterms:format":[{"property_id":9,"type":"literal","is_public":"1","@language":"","@value":"JPEG Image"}]}
