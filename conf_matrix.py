import sys

def parseline(line):
    ls = line.split('\t')
    pred = int(ls[0])
    label = int(ls[1])
    corstr = ls[2]
    if (pred==label and corstr == "correct\n") or (pred!=label and corstr == "incorrect\n"):
        return pred, label
    else:
        return None, None


def checkconfm(confm, acc):
    corrects = 0
    total = 0
    for k in confm.keys():
        corrects += confm[k][k]
        for lk in confm[k].keys():
            total += confm[k][lk]
    pacc = float(corrects)*100/total
    assert(abs(pacc -acc)<0.01)

def writeconfm(confm, sfile):
    with open("confmatrices/"+ sfile +".log", "w") as sf:
        sf.write("\\begin{table}[]\n")
        sf.write("\\centering\n")
        sf.write("\\caption{My-Caption}\n")
        sf.write("\\label{My-Label}\n")
        sf.write("\\begin{tabular}{llllllll}\n")
        sf.write("\t & \\multicolumn{7}{1}{labels}")
        sf.write("\t & anger & contempt & disgust & fear & happiness & sadness & surprise \\\\")

def main():
    if len(sys.argv) != 2:
        exit(0)
    fname = sys.argv[1]
    print "Processing file:{}".format(fname)
    with open(fname, "rb") as cmfile:
        confm = {}
        for i in range(7):
            confm[i] = {}
            for j in range(7):
                confm[i][j] = 0

        for line in cmfile.readlines():
            if "acc:" not in line:
                pred, label = parseline(line)
                confm[pred][label] += 1
            else:
                acc = float(line.split(':')[1])

        checkconfm(confm, acc)
        writeconfm(confm, fname.split('.')[0])



if __name__== '__main__':
    main()