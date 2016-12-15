import pandas
import numpy as np
import random
import string

def random_string(size=7):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(size))


def synthetic_schemas(csv_file, number_of_schemas=5, frac=.01):
    data = pandas.read_csv(csv_file,dtype=str)
    columns = list(data.columns)
    mapping_list = []
    schema_list = []

    for _ in range(number_of_schemas):
        # constructing the schema
        schema = data.sample(frac=frac)
        permutation = np.random.permutation(columns)
        schema = schema.ix[:,permutation]
        schema.columns = [random_string() for _ in range(len(columns))]
        schema_list.append(schema)
        # constructing the mapping
        mapping = dict(zip(schema.columns, permutation))
        mapping_list.append(mapping)

    return (schema_list, mapping_list)


def main():
    (schema_list, mapping_list) = synthetic_schemas('movie_metadata.csv', 2)
    print schema_list
    print mapping_list


if __name__ == "__main__":
    main()
