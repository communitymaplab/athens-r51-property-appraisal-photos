import pandas as pd

Bradberry_1962_photos = pd.read_csv("data/Bradberry appraisals 1962/processed_photos.csv")
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
# add prefix to path
Diaz_1964_photos["path"] = "https://raw.githubusercontent.com/communitymaplab/athens-r51-property-appraisal-photos/refs/heads/main/pipeline/data/Diaz%20appraisals%201964/processed/" + Diaz_1964_photos["path"]

merged_photos = pd.concat([Bradberry_1962_photos, Diaz_1964_photos])

merged_photos = merged_photos.assign(parcel=merged_photos['parcel'].str.split(',')).explode('parcel')

merged_photos['block'] = 'B' + merged_photos['block'].astype(str)
merged_photos['parcel'] = 'P' + merged_photos['parcel'].astype(str)

merged_photos.to_csv("data/merged_photos.csv", index=False)