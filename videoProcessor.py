import numpy as np
import cv2 
import sys, getopt

def main(argv):
	# Test for appropriate input/output syntax
	inputfile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
	except getopt.GetoptError:
		print 'Error: Use format videoProcessor.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'videoProcessor.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	print 'Input file is :', inputfile
	print 'Output file is :', outputfile
	
	# Take in the input file
	cap = cv2.VideoCapture(inputfile)
	fourcc = cv2.cv.CV_FOURCC(*'DIV4')
	fgbg = cv2.BackgroundSubtractorMOG()
#	out = cv2.VideoWriter(outputfile, cap.get(6), cap.get(5), (cap.get(int(3)),cap.get(int(4))))
	i = 1
	while(cap.isOpened()):
		ret, frame = cap.read()
		if i % 200 == 0 :
			if ret == True:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				eqhist = cv2.equalizeHist(gray)
#				fgmask = fgbg.apply(eqhist)  
				cv2.imshow('frame',eqhist)			
#				out.write(gray)
			
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		i += 1

	cap.release()
#	out.release()
	cv2.destroyAllWindows()

# Execute only if run as a script
if __name__ == "__main__":
	main(sys.argv[1:])
