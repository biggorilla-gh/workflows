from flexmatcher import FlexMatcher
import schemas

def main():
    (schema_list, mapping_list) = schemas.synthetic_schemas('movie_metadata.csv')
    flexmatcher = FlexMatcher()
    flexmatcher.create_training_data(schema_list, mapping_list)
    flexmatcher.train()

    # Making a prediction for a new data source
    (test_schema_list, test_mapping_list) = schemas.synthetic_schemas('movie_metadata.csv', 1, 0.01)
    test_schema = test_schema_list[0]
    test_mapping = test_mapping_list[0]

    predicted_mapping = flexmatcher.make_prediction(test_schema)
    matches = 0
    for k in test_mapping:
        if test_mapping[k] == predicted_mapping[k]: matches += 1
        print test_mapping[k], ' ---> ', predicted_mapping[k]
    print '----------'
    print matches


if __name__ == "__main__":
    main()
