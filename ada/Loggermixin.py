import logging
import logging.config
import os
import sys
from logging import Logger

class Loggermixin(object):
	# Logging.conf location
	#Preferred location is alongside Loggermixin class in the same folder

	logging_conf_filepath = os.path.join(os.path.dirname(__file__), 'logging.conf')

	def __init__(self):
		self._logger=None

	@property
	def logger(self):
		if self._logger:
			return self._logger
		else:
			raise ValueError('Loggermixin- Logger is not set!')

	@logger.setter
	def logger(self, value):
		if not value:
			raise ValueError('Logger is not initialized')
		elif not isinstance(value, Logger):
			raise ValueError('value is of type ', str(type(value)), ' must be of type Logger')
		else:
			self._logger = value

	# this logger intended for use with helper functions outside of a class.
	#
	# usage example:
	# 	from Loggermixin import *
	#
	# 	h_logger=Loggermixin.get_default_logger()
	#
	#   h_logger.debug('Some log message')
	#
	@classmethod
	def get_default_logger(cls):
		if os.path.exists(Loggermixin.logging_conf_filepath):
			logging.config.fileConfig(Loggermixin.logging_conf_filepath)
		else:
			print('logging.conf file not found at path '+ Loggermixin.logging_conf_filepath)
		if ('run.py' in sys.argv[0]):
			h_logger=logging.getLogger('runner')
		elif('train.py' in sys.argv[0]):
			h_logger=logging.getLogger('trainer')
		else:
			h_logger=logging.getLogger('other')

		return h_logger
