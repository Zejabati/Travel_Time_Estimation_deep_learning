import datetime

class Logger:
    def __init__(self, exp_name):
        #self.file = open('C:/Users/mostafa3/OneDrive - University of Manitoba/Desktop/TTE/DeepTTE-master_T9/log.log'.format(exp_name), 'w')
        self.file = open('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/TTE final model/logs/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()



