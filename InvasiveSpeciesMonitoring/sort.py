import re
class Sort:
	def __init__(self):
		self.value = 0

	def numericalSort(self,value):
		numbers = re.compile(r'(\d+)')
		parts = numbers.split(value)
		parts[1::2] = map(int, parts[1::2])
		return parts

