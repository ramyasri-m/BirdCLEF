# BirdCLEF

1. Clone this repository:

```
git clone git@github.com:jsalbert/sound_classification_ml_production.git
```
2. Create a virtual environment using [virtualenv](https://virtualenv.pypa.io/en/latest/) and install library requirements:

```
pip install virtualenv
virtualenv myenv
source myenv/Scripts/activate #For Windows
pip install -r requirements.txt
```

3. Go to the folder `flask_app` and run the app locally in your computer:

```
python app.py 
```

4. Access it via [localhost:5000](http://localhost:5000/)

You should be able to see this screen, upload and classify a sound:
