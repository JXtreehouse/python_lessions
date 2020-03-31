import glob
import gzip


# unzip .gz file and cat 1 text file  =============================
def unzip_generate_txt(filePath):
    # filePath = "./my_log.txt"

    files_file = glob.glob("./zc*.gz")  # wildcard: "app" and ".gz"
    print(files_file)

    with open(filePath, 'wb') as saveFile:
        for file in files_file:
            print(file)

            f = gzip.open(file, 'rb')
            data = f.read()
            saveFile.write(data)
        saveFile.flush()
        f.close()


# find maximum line number gap between "onStop"  ==================
#
# Assumption1 : there are some "onStop" line before recording start (more than 2 lines)
# Assumption2 : Recording part is, maximum line number gap between "onStop" line.
#
def find_max_onStop_gap(filePath):
    ld = open(filePath)
    lines = ld.readlines()
    ld.close()
    count = 0
    count_detect = 0
    count_pre = 0

    line_list = []
    diff_list = []

    for line in lines:

        ad = line.find("onStop()")
        if ad >= 0:

            if count_detect > 0:
                diff_list.append(count - count_pre)
                line_list.append(count_pre)

            count_detect = count_detect + 1
            count_pre = count

        count = count + 1

    index = diff_list.index(max(diff_list))
    firstOnStop = line_list[index]

    print(diff_list)
    print(line_list)
    print("fisrt onStop line", firstOnStop)

    return firstOnStop


# skip data until "first OnStop line" and write data ==================
def skip_onStop_generate_txt(srcfilePath, filePath, firstOnStop):
    ld = open(srcfilePath)
    lines = ld.readlines()
    ld.close()

    # filePath = "./my_log_crop.txt"
    count = 0
    with open(filePath, 'w') as saveFile:
        for line in lines:

            # print(count)
            count = count + 1

            if count > firstOnStop:
                saveFile.write(line)
            saveFile.flush()

    print("generated, ", filePath)


if __name__ == '__main__':
    # unzip .gz file. and generate 1 text file
    genfn = "./my_log.txt"
    unzip_generate_txt(genfn)

    # find "onStop" line number, and find max gap line
    firstonStop = find_max_onStop_gap(genfn)

# generate text file
# genfn_skip = "./all_log_skip.txt"
# skip_onStop_generate_txt(genfn, genfn_skip, firstonStop)
