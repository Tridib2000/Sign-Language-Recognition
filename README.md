*This code is created and and run in kaggle.
*if it is required to run in a local machine, the directories must be changed.
code-
      csv_directory = r'/kaggle/input/sign-language-digits/train'
      labels_file = r'/kaggle/input/labels-for-numbers/labels.csv'

*for extraction of landmarks for iamges, they should be put in a folder and in registration_mediapipe python file a's value should be the name of the folder. A train folder have to be created where the exracted landmarks will be saved in csv format.

*in order to run the model, all files from model and the test run file should be put together and directories must be updated
with open(r'model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model's weights
model.load_weights(r'model_weights.h5')
