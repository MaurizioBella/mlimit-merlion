# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""main script"""
import warnings
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from flask_cors import CORS
import src.config as config
from src.web.default import default_app
from src.api.config import config_route


warnings.filterwarnings("ignore")

DB = SQLAlchemy()


def create_app():
    '''
    used by gunicorn wsgi
    '''
    app = Flask(__name__, static_url_path='/',
                static_folder='static',
                template_folder='templates')

    CORS(app, resources={r"*": {"origins": "*"}})
    app.config.from_object(config.APP_SETTINGS)  # src.config.DevelopmentConfig
    DB.init_app(app)

    app.secret_key = config.SECRET_KEY

    # Factor the application into a set of blueprints
    app.register_blueprint(default_app)  # default or home blueprint
    # used to enable/disable configuration measures
    app.register_blueprint(config_route, url_prefix='/api')

    return app
