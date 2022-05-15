# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Init db migration
# export FLASK_APP=src; flask db init; flask db migrate -m "Initial migration."
# FLASK_APP=src flask db upgrade
# FASK_ENV=development flask run -h localhost -p 3000
gunicorn wsgi:"create_app()"