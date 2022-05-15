# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""used by gunicorn"""
from src import create_app

if __name__ == "__main__":
    app = create_app()
