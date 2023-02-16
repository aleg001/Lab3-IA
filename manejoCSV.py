
import csv

def TxtReaderWriter(txt_file, csv_file):
    with open(txt_file, "r",) as in_text:
        in_reader = csv.reader(
            in_text,
            quotechar='"',
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True,
        )
        with open(csv_file, "w") as out_csv:
            out_writer = csv.writer(out_csv)
            out_writer.writerow(["spam", "message"])
            for row in in_reader:
                out_writer.writerow(row)

