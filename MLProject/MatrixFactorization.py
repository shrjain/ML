from numpy import *
path_prefix = '/Users/shrainik/Downloads/download/training_set/'

if __name__ == '__main__':
    numberOfMovies = 101
    movieRatings = []
    for i in range(1, numberOfMovies):
        movieRatings.append(genfromtxt(path_prefix + 'mv_'+'{0:07d}'.format(i) + '.txt', dtype = [('userId','i'), ('rating','i'), ('date','S10')] ,delimiter=',', skiprows = 1))
    
    print len(movieRatings)