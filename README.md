# Facial Recognition and Authentication : Python

This is a facial recognition and authentication system in python.

This script uses OpenCV to detect a face and then compares that image with the images provided in the dataset.
Since as an output we only require single enitiy(ie name of the person), This script captures 30 frames consecutively and recognises each and every one of those frames. The result that comes out the maximum number of times is returned as an output.

If the person is not specified in the dataset, the script returns an error informing about not recognizing the user.

It is recommemded to use proper lighting while using this script. 
