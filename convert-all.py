import sys
import os
import pandas as pd


excel_file = sys.argv[1]

xls = pd.ExcelFile(excel_file)

outdir = excel_file.split(".")[0]
os.makedirs(outdir, exist_ok=True)

for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.fillna(0, inplace=True)
    df = df.astype(int)
    outname = sheet_name.replace(" ", "_").lower().encode('ascii', 'ignore').decode()
    df.to_csv(f'{outdir}/{outname}.csv', index=False, header=False)
    print(f'Saved {outdir}/{outname}.csv')
