# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class ValueTooSmallError(Error):
    """Exception raised for errors in the input salary.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Value too small. MLimit cannot be used when number of rows in your data is less than MERLION_LIMIT_EVENTS"):
        self.message = message
        super().__init__(self.message)


class NoSalesforceConnection(Error):
    """Exception raised if there is no connection with Salesforce.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Connection Issue. There is no connection with a Salesforce Organization"):
        self.message = message
        super().__init__(self.message)
