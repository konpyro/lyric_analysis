import lyrics
import csv

wfile = open('MoodyLyricsFullSmall4Q.csv', 'w')
csv_writer = csv.writer(wfile, delimiter=",")

with open('MoodyLyrics4Q.csv') as rfile:
    csv_reader = csv.reader(rfile, delimiter=",")

    count = 0
    for row in csv_reader:
        try:
            new_row = [row[0], row[1], row[2], row[3], lyrics.getlyrics(row[1], row[2], False)]
        except:
            new_row = [row[0], row[1], row[2], row[3], ' ']
        csv_writer.writerow(new_row)
        count += 1
        if count == 11:
            break

# ly = lyrics.getlyrics("Eminem", "Stan", False)
# print(ly)
