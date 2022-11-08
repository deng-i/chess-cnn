import sys
from segmentpic import segment
from cnn import predict_img
from PIL import Image

filename = ""
if len(sys.argv) != 2:
    print("Give filename as argument")
    pass
else:
    filename = sys.argv[1]
tiles2d = segment(filename)
i = 0
for tiles in tiles2d:
    for tile in tiles:
        im = Image.fromarray(tile)
        savename = "tiles/" + str(i) + ".jpg"
        im.save(savename)
        i += 1

predictions = []
for i in range(64):
    fn = "tiles/" + str(i) + ".jpg"
    predictions.append(predict_img(img_path=fn))
fen_dict = {"br" : "r",
            "bb" : "b",
            "bn" : "n",
            "bq" : "q",
            "bk" : "k",
            "bp" : "p",
            "wr": "R",
            "wb": "B",
            "wn": "N",
            "wq": "Q",
            "wk": "K",
            "wp": "P",
            "empty": ""}
fen = ""
for i in range(8):
    k = 0
    for j in range(8):
        if predictions[i * 8 + j] == "empty":
            k += 1
        else:
            if k != 0:
                fen += str(k)
                k = 0
            fen += fen_dict[predictions[i * 8 + j]]
    if k != 0:
        fen += str(k)
        k = 0
    fen += "/"
fen = fen[:-1]
print(fen)