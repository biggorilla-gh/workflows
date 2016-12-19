#!/usr/bin/python
"""
Authored by Wang-Chiew Tan
"""
import re
import json
import sys

# remove all html tags, new line characters within the extracted paragraphs <p>
def cleanhtml(extracted):
    # remove html tags
    cleantext = re.sub('<.*?>', '', extracted)
    # remove all new lines
    cleantext = re.sub('\n *', "", cleantext)
    return cleantext


def main():
        ifilename = str(sys.argv[1])
        ofilename = str(sys.argv[2])
        with open(ofilename, 'w') as outfile:
            with open(ifilename, 'r') as ifile:
                for json_line in ifile:
                    data = json.loads(json_line)
                    newdata = []
                    for s in data["content"]:
                        newdata.append(cleanhtml(s))
                    data["content"] = newdata
                    json.dump(data, outfile)

if __name__ == "__main__":
    main()
