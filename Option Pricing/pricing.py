import numpy as np
import random as rand
from optparse import OptionParser, OptionGroup
from scipy.stats import norm

# defaults
call = True
put = False
european = True
asian = False
barrier = None

S0 = 100.
r = 0.05
sig = 0.20
T = 1.
E = 100.

arithmetic = True
geometric = False

M = 100 # no. of sample paths
N = 100 # no. of time steps

up = True
down = False
In = True
out = False

parser = OptionParser()
group = OptionGroup(parser, "Option Types", "Select which types of options to use. " 
			"Default is European Call")
group.add_option("-c", "--call", action="store_true", dest="call", help="Select for a call option")
group.add_option("-p", "--put", action="store_true", dest="put", help="Select for a put option")
group.add_option("-e", "--european", action="store_true", dest="european", help="Select for a European option")
group.add_option("-a", "--asian", action="store_true", dest="asian", help="Select for an Asian option")
group.add_option("-b", "--barrier", action="store", dest="barrier",
				 help="Select for an Barrier option. Enter amount for barrier here.")
group.add_option("-u", "--up", action="store_true", dest="up", help="Barrier: up")
group.add_option("-d", "--down", action="store_true", dest="down", help="Barrier: down")
group.add_option("-i", "--in", action="store_true", dest="In", help="Barrier: in")
group.add_option("-o", "--out", action="store_true", dest="out", help="Barrier: out")
parser.add_option_group(group)

group = OptionGroup(parser, "Initial Data", 
			"Set initial parameters for the Monte Carlo method. "
			"Defaults: S0 = 100, r = 0.05, sigma = 0.20, T = 1, E = 100")
group.add_option("-S", "--S0", action="store", dest="S0", help="Initial value of asset / spot")
group.add_option("-r", "--IR", action="store", dest="r", help="Interest Rate")
group.add_option("-s", "--sigma", action="store", dest="sig", help="Volatility")
group.add_option("-T", "--expiry", action="store", dest="T", help="Time at expiry (years)")
group.add_option("-E", "--strike", action="store", dest="E", help="Strike price")
parser.add_option_group(group)

group = OptionGroup(parser, "Averages", "Select which type of average to use. " 
			"Default is Arithmetic average")
group.add_option("-A", "--arithmetic", action="store_true", dest="arithmetic", help="Arithmetic average")
group.add_option("-G", "--geometric", action="store_true", dest="geometric", help="Geometric average")
parser.add_option_group(group)

group = OptionGroup(parser, "Iteration parameters", 
			"Set number of sample paths M and number of time steps N. "
			"Defaults: M = 100, N = 100")
group.add_option("-M", "--M", action="store", dest="M", help="Number of sample paths")
group.add_option("-N", "--N", action="store", dest="N", help="Number of time steps")
parser.add_option_group(group)

group = OptionGroup(parser, "Graph options", "Options for displaying an example graph. ")
group.add_option("-g", "--graph", action="store_true", dest="graph",
				 help="Graph of option values, with theoretical for Europeans")
parser.add_option_group(group)

(options, args) = parser.parse_args()

# apply args
if options.european:
	european = True
	asian = False

elif options.asian:
	asian = True
	european = False
	
else:
	if options.barrier!=None:
		barrier = float(options.barrier)
		european = False
		asian = False
	
	if options.up:
		asian = False
		european = False
		up = True
		down = False
		
	elif options.down:
		asian = False
		european = False
		up = False
		down = True
	
	if options.In:
		asian = False
		european = False
		In = True
		out = False
		
	elif options.out:
		asian = False
		european = False
		In = False
		out = True

if options.call:
	call = True
	put = False

elif options.put:
	put = True
	call = False

if options.S0 != None:
	S0 = float(options.S0)

if options.r != None:
	r = float(options.r)

if options.sig != None:
	sig = float(options.sig)

if options.T != None:
	T = float(options.T)

if options.E != None:
	E = float(options.E)
	
if options.arithmetic:
	arithmetic = True
	geometric = False

elif options.geometric:
	geometric = True
	arithmetic = False
	
if options.M != None:
	M = int(options.M)

if options.N != None:
	N = int(options.N)
	
if options.graph:
	graph = True
else:
	graph = False
	
# Print parameters
string = ""
if european: string += "European "
elif asian: string += "Asian "
elif barrier != None:
	string = "Barrier "
	if up:
		string += "Up "
	elif down:
		string += "Down "
	if In:
		string += "and In "
	elif out:
		string += "and Out "

if call: string += "Call "
elif put: string += "Put "
	
string += "option"
if barrier!=None: string += ". Barrier = "+str(barrier)
print string

string = "S0 = " + str(S0) + "; r = " + str(r) + "; sigma = " + str(sig)
string += "; T = " + str(T) + "; E = " + str(E)
print string

if arithmetic: print "Arithmetic average"
elif geometric: print "Geometric average"

if graph:
	print "Producing a graph taking sample values M = 100, 1000, 10000, 100000. N = " + str(N)
if not graph:
	print "M = " + str(M) + " sample paths; N = " + str(N) + " time steps"

# Setup functions

def AA(S):
	""" arithmetic average """
	A = 0
	l = np.size(S,0)
	for i in range(l):
		A += S[i]
	A = A/l
	return A
	
def GA(S):
	""" geometric average """
	A = 0
	l = np.size(S,0)
	for i in range(l):
		A += np.log(S[i])
	A = np.exp(A/l)
	return A
	

def PV(S):
	""" calculate the present value of the option """
	l = np.size(S,0)
	if european:
		if call:
			V = np.exp(-r*T)*max([S[l-1] - E, 0])
		elif put:
			V = np.exp(-r*T)*max([E - S[l-1], 0])
			
	elif asian:
		if arithmetic:
			A = AA(S)
		elif geometric:
			A = GA(S)
		if call:
			V = np.exp(-r*T)*max([S[l-1] - A, 0])
		elif put:
			V = np.exp(-r*T)*max([A - S[l-1], 0])
			
	elif barrier!=None:
		if In:
			V = 0
			if up:
				for i in range(l):
					if S[i] > barrier:
						if call: V = np.exp(-r*T)*max([S[l-1]-E,0])
						if put: V = np.exp(-r*T)*max([E-S[l-1],0])
						break
			elif down:
				for i in range(l):
					if S[i] < barrier:
						if call: V = np.exp(-r*T)*max([S[l-1]-E,0])
						if put: V = np.exp(-r*T)*max([E-S[l-1],0])
						break
		elif out:
			if call: V = np.exp(-r*T)*max([S[l-1]-E,0])
			if put: V = np.exp(-r*T)*max([E-S[l-1],0])
			if up:
				for i in range(l):
					if S[i] > barrier:
						V = 0
						break
			if down:
				for i in range(l):
					if S[i] < barrier:
						V = 0
						break		
		
	return V

# Simulate the asset sample paths
# if graph mode used, take many values of M
if graph:
	Mlist = np.array([100,1000,10000,100000])
	Y = np.zeros(np.size(Mlist,0))
else:
	Mlist = [M]

count = -1
# taking the number of sample paths from a list
for M in Mlist:
	dt = T/N
	sdt = np.sqrt(dt)
	
	S = np.zeros((M,N+1))
	V = np.zeros(M)
	
	count += 1

	value = 0.
	
	""" simulation of sample paths is here """
	for i in range(M):
		S[i,0] = S0
		for j in range(N):
			phi = rand.gauss(0,1)
			S[i,j+1] = S[i,j]*(1 + r*dt + sig*phi*sdt)
		# Calculate the discounted payoff of each sample path
		V[i] = PV(S[i,:])
		value += V[i]
	
	# Calculate the value of the option, print
	value = value/M
	if graph:
		Y[count] = value
	else:
		print "Simulated value = " + str(value)
	
	# If using a European option, compare with the theoretical Black-Scholes price
	if european:
		d1 = (np.log(S0/E)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T))
		d2 = d1 - sig*np.sqrt(T)
		if call:
			ans = S0 * norm.cdf(d1) - E * np.exp(-r*T) * norm.cdf(d2)
		elif put:
			ans = E * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
		if not graph: print "Theoretical value = " + str(ans)
			
	elif barrier!=None:	
		""" get the theoretical values for the barrier options,
			using Haug's method """
		if call:
			phi = 1
		elif put:
			phi = -1
		if down:
			eta = 1
		elif up:
			eta = -1
		
		mu = (r-0.5*sig**2)/(sig**2)
		lam = np.sqrt(mu**2+(2*r)/(sig**2))
		
		ssT = sig*np.sqrt(T)
		
		x1 = np.log(S0/E)/ssT + (1+mu)*ssT
		y1 = np.log(barrier**2/(S0*E))/ssT + (1+mu)*ssT
		x2 = np.log(S0/barrier)/ssT + (1+mu)*ssT
		y2 = np.log(barrier/S0)/ssT + (1+mu)*ssT
		z = np.log(barrier/S0)/ssT + lam*ssT
		
		A = phi*S0*norm.cdf(phi*x1) - phi*E*np.exp(-r*T)*norm.cdf(phi*x1-phi*ssT)
		B = phi*S0*norm.cdf(phi*x2) - phi*E*np.exp(-r*T)*norm.cdf(phi*x2-phi*ssT)
		C = phi*S0*(barrier/S0)**(2*(mu+1))*norm.cdf(eta*y1) - phi*E*np.exp(-r*T)*(barrier/S0)**(2*mu)*norm.cdf(eta*y1-eta*ssT)
		D = phi*S0*(barrier/S0)**(2*(mu+1))*norm.cdf(eta*y2) - phi*E*np.exp(-r*T)*(barrier/S0)**(2*mu)*norm.cdf(eta*y2-eta*ssT)
		
		if (E > barrier and down) and (In and call): ans = C
		if (E > barrier and down) and (out and call): ans = A-C
		if (E > barrier and up) and (In and call): ans = A
		if (E > barrier and up) and (out and call): ans = 0
		if (E > barrier and down) and (In and put): ans = B-C+D
		if (E > barrier and down) and (out and put): ans = A-B+C-D
		if (E > barrier and up) and (In and put): ans = A-B+D
		if (E > barrier and up) and (out and put): ans = B-D
		
		if (E < barrier and down) and (In and call): ans = A-B+D
		if (E < barrier and down) and (out and call): ans = B-D
		if (E < barrier and up) and (In and call): ans = B-C+D
		if (E < barrier and up) and (out and call): ans = A-B+C-D
		if (E < barrier and down) and (In and put): ans = A
		if (E < barrier and down) and (out and put): ans = 0
		if (E < barrier and up) and (In and put): ans = C
		if (E < barrier and up) and (out and put): ans = A-C
		
		if not graph: print "Theoretical value = " + str(ans)
		
if graph:
	# plotting a log graph
	import matplotlib.pyplot as plt
	title = "Simulated value"
	yaxis = "$V_M$"
	if european or barrier!=None:
		yaxis = "$e_{M}$"
		title = "Error plot"
		Y = Y - ans
	plt.semilogx(Mlist, Y, linestyle = '-.', c = 'c', marker = 'x', mec = 'b')
	plt.axhline(y=0, linestyle = '--', c = 'g')
	if european or barrier!=None: plt.ylim(-1.5,1.5)
	plt.xlabel("$M$")
	plt.ylabel(yaxis)
	plt.title(title)
	plt.show()
		