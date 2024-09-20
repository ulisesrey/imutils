# Installation
To install this as package you should be familiar with the process. But briefly these are the main steps using anaconda:

1. Open Terminal and activate your environment (example your_env)
	```
	conda activate your_env
	```
2. Install it with a pip command (pointing to the directory where you cloned it)
	```
	pip install /code/imutils/
	```
	or better:
	cd to the directory where you cloned the directory
	```
	cd ../code/imutils/
	```
	And then pip install the local directory
	```
	pip install .
	```
	
	This would be wrong:
	```
	pip install imutils
	```
	Because it will install another imutils package that someone has uploaded to pip.



This package has some image processing tools written by Ulises.
