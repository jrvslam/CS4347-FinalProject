# Backend

This folder contains the code necessary for running the backend server.

The `melody` folder contains code duplicated from the `melody_extraction` folder, with all the code necessary for running the pretrained model.

## Starting the backend server

First, install all required dependencies:
```
pip install -r requirements.txt
```

To start the server in development mode:
```
flask --app app run
```

To start the server in production mode:
```
python app.py
```