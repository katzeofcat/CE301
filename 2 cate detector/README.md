This detector only for 2 categories

Input is to prepare the data for training. In this project I use the CAT VS DOG data set from this link:

https://www.microsoft.com/en-us/download/details.aspx?id=54765

If want to train different dataset, some detail in input.py must be change like the categories, image direction (which image is belong to which categories)

They will generate .pickle data which is use for training. X is for data input and Y is for label input

For Training just change the input to corresponding data name.

In the end the train will generate a .h5 file change the model (model = tf.keras.models.load_model("md.h5")) in predict.py to the corresponding .h5 file
and the image to the corresponding image directory to run the prediction.