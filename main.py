import sys
from features_extraction.dataset import Dataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path_to_xml_file')
        exit(1)

    # create dataset frame
    dataset = Dataset(filename=sys.argv[1])

    # add emoji counts
    dataset.add_emoji_count('all')

    # add emoji type counts
    dataset.add_emoji_type_count('all')

