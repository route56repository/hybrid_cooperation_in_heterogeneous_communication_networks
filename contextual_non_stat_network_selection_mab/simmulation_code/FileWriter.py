__author__ = 'lekesen'

# Class to save the obtained results in files.

class FileWriter:

	def __init__(self, scenario):
		self.f_results = open("../results/results_muMAB", 'w')
		self.f_ba = open("../results/ba_muMAB", 'w')
		self.f_time = open("../results/time_muMAB", 'w')
		self.scenario = scenario

	def write_settings(self, strategies):
		# Settings for results
		self.f_results.write("SETTINGS\n")
		self.f_results.write("m: " + str(self.scenario.m) + "\n")
		self.f_results.write("NRuns: " + str(self.scenario.NRuns) + "\n")
		self.f_results.write("NSteps: " + str(self.scenario.NSteps) + "\n")
		self.f_results.write("Strategy:")
		
		for strategy in strategies:
			self.f_results.write(" " + strategy)
		
		self.f_results.write("\nSTART\n")

		# Settings for best-action
		self.f_ba.write("SETTINGS\n")
		self.f_ba.write("m: " + str(self.scenario.m) + "\n")
		self.f_ba.write("NRuns: " + str(self.scenario.NRuns) + "\n")
		self.f_ba.write("NSteps: " + str(self.scenario.NSteps) + "\n")
		
		self.f_ba.write("Strategy:")
		for strategy in strategies:
			self.f_ba.write(" " + strategy)

		self.f_ba.write("\nSTART\n")

		# Settings for time
		self.f_time.write("SETTINGS\n")
		self.f_time.write("m: " + str(self.scenario.m) + "\n")
		self.f_time.write("NRuns: " + str(self.scenario.NRuns) + "\n")
		self.f_time.write("NSteps: " + str(self.scenario.NSteps) + "\n")
		
		self.f_time.write("Strategy:")
		for strategy in strategies:
			self.f_time.write(" " + strategy)

		self.f_time.write("\nSTART\n")

	def write_results(self, strategy, avg_regret, ba, time):
		self.f_results.write("Strategy: " + strategy + "\n")
		for item in avg_regret:
			self.f_results.write("%s " %item)
		self.f_results.write("\n")

		self.f_ba.write("Strategy: " + strategy + "\n")
		for item in ba:
			self.f_ba.write("%s " %item)
		self.f_ba.write("\n")

		self.f_time.write("Strategy: " + strategy + "\n")
		self.f_time.write("%s " %time)
		self.f_time.write("\n")

	def finish(self):
		self.f_results.write("FINISH")
		self.f_results.close()

		self.f_ba.write("FINISH")
		self.f_ba.close()

		self.f_time.write("FINISH")
		self.f_time.close()

		# Create file to show that it has finished
		f_finished = open("../results/FINISHED", 'w')
		f_finished.write("DONE")
		f_finished.close()