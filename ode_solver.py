import numpy as np
import matplotlib.pyplot as plt
import copy

#%%
### The algorithm creates a total of npop solutions in each generation. The next generation is created
### by breeding new solutions from parents selected from the previous generation. Solutions with
### a better fit are more likely to be selected as parents, and each couple creates two new children.
### nwinners defines the number of solutions in each generation that are kept in the new generation
### - better fitting solutions are more likely to be kept, and the best solution is always kept
### The remaining solutions are made up of new randomly generated solutions.
npop=55
ncouples=20
nwinners=10

nseg=8 # nseg is the number of points in each candidate solution
nmax=1000000 # number of options avaiable for each point
ymax=1 # max expected y value in the solution

mutation_prob=0.2 # mutation rate for each point in a child
mutation_amount=0.05 #percent of ymax that points are shifted when mutated
incout=50 # every incout iteration is graphed
total_iterations=1000 # number of iterations the algorithm runs for
repeatable=False # if true, a random number seed is chosen

restrict_sol=1 # set to < 0 for negative only solution, > 0 for positive only
               # solutions and 0 for both positive and negative solutions

### Define boundary conditions. If boundary values are not known, set equal
### to 'None'. Algorithm requires at least two boundary values to solve for
### a specific solution.
x0=0
xN=5

y0=1
y0_prime=-1

yN=0.00673
yN_prime=None

### Define fitness function weightings
### A1=sumfit, A2=worst, A3=prod, A4=bval, A5=bval_grad
A1, A2, A3, A4, A5 = 5, 1, 1, 5, 1

### Define ODE - cc2*y" + cc1*y' + cc0*y = 0
c2, c2x, c2xs = 1, 0, 0 #cc2 = c2 + c2x*x + c2xs*x^2
c1, c1x, c1xs = 0, 0, 0 #cc1 = c1 + c1x*x + c1xs*x^2
c0, c0x, c0xs = -1, 0, 0 #cc0 = c0 + c0x*x + c0xs*x^2

### Add expected curve to compare results
x_exp=np.linspace(x0,xN,10*nseg)
y_exp=np.exp(-x_exp)

#%%
def convert(candidate):
    ### Converts the integers in each candidate into the y values which
    ### can be used to assess the fitness and be graphed.
    ### Calculated by y=m*eps, where m is an integer in the candidate

    y=[]
    for n in range(len(candidate)):
        y.append(candidate[n]*ymax/nmax)

    return y

#----------------

def graph(x,y,iteration,best_weight,x_exp,y_exp):
    ### Graphs a candidate solution using the y values in convert.
    ### Also plots the expected curve to compare with the current solution.

    fit = abs(y-y_exp[::10])/nseg
    total_fit = sum(fit)
    # fig = plt.figure()

    # ax1.clear()


    # ax1, = plt.plot([], [])

    #plt.scatter(x,y,marker='x',color='red')
    plt.ion()
    plt.cla()
    plt.scatter(x,y,marker='x',color='red')
    plt.plot(x,y)
    plt.plot(x_exp,y_exp,color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('number of iterations = {0:}'.format(iteration))
    plt.annotate('fit to curve = {0:.3g}'.format(total_fit),(2*x_exp[-1]/3,max(y_exp)/3))
    plt.annotate('best weight={0:.3g}'.format(best_weight),(2*x_exp[-1]/3,max(y_exp)/2))
    plt.draw()
    plt.pause(0.001)



    # plt.show()

#----------------

def fitness(y):
    ### Uses y values defined in convert to assess the fitness.

    x=np.linspace(x0,xN,nseg)

    fit=np.zeros(nseg)
    dx=(xN-x0)/(nseg-1)
    d1=2*dx
    d2=dx**2
    sumfit=0
    grad1=0
    grad2=0
    bval1=0
    bval2=0

    #calculate the fit of each point - measure of how far the point strays
    # from  a solution to the ODE, using central difference
    #for interior points:
    for n in range(nseg):

        cc2=c2+x[n]*(c2x+c2xs*x[n])
        cc1=c1+x[n]*(c1x+c1xs*x[n])
        cc0=c0+x[n]*(c0x+c0xs*x[n])

        if n==0:
            fitc = (cc2/d2-3*cc1/d1+cc0)*y[1] + \
                (-2*cc2/d2+4*cc1/d1)*y[2]+(cc2/d2-cc1/d1)*y[3]
            fit[0] = abs(fitc)

        elif n==nseg-1:
            fitc = (cc2/d2+cc1/d1)*y[-3]+(-2*cc2/d2-4*cc1/d1)*y[-2] + \
                (cc2/d2+3*cc1/d1+cc0)*y[-1]
            fit[-1] = abs(fitc)

        else:
            fitc=(cc2/d2+cc1/d1)*y[n+1]+(-2*cc2/d2+cc0)*y[n] + \
                (cc2/d2-cc1/d1)*y[n-1]
            fit[n]=abs(fitc)

    #sum of the fit values
    for n in range(0,nseg):
        sumfit=sumfit+abs(fit[n])/nseg

    #product of fit values
    prod=np.prod(fit)

    #max fit in a solution
    worst = np.max(fit)

    #calculate gradients at the boundaries
    glhs=(-3*y[1]+4*y[2]-y[3])/d1
    grhs=(y[-3]-4.0*y[-2]+3.0*y[-1])/d1

    #only use boundary values if they are known
    if y0_prime!=None and c2!=0:
        grad1=abs(glhs-y0_prime)

    if yN_prime!=None and c2!=0:
        grad2=abs(grhs-yN_prime)

    if y0!=None:
        bval1=abs(y[0]-y0)

    if yN!=None:
        bval2=abs(y[-1]-yN)

    bval_grad=grad1+grad2
    bval=bval1+bval2

    #total fitness
    weight= A1*sumfit + A2*worst + A3*prod + A4*bval + A5*bval_grad

    return weight


def score_population(population):
    ### Asses fitness of every candidate in the population
    weights=[]
    y = [convert(population[k]) for k in range(len(population))]

    weights = [fitness(i) for i in y]

    return weights

#----------------

def pick_parent(weights):
    ### Picks solutions to become parents. Solutions with a better fitness
    ### are more likely to be picked

    array=np.array(weights)
    temp=array.argsort()
    ranks=np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    score = [len(ranks) - x for x in ranks]
    cum_scores = copy.deepcopy(score)
    for i in range(1,len(cum_scores)):
        cum_scores[i] = score[i] + cum_scores[i-1]
        probs = [x / cum_scores[-1] for x in cum_scores]
        rand = np.random.random()

    for i in range(0, len(probs)):
        if rand < probs[i]:

            return i

#-----------------

def breed(p1,p2):
    ### New solutions are bred by selecting a random section along the parents
    ### and swapping these between them. Creates two new children.

    n=np.random.randint(nseg)
    c1=np.append(p1[:n],p2[n:])
    c2=np.append(p2[:n],p1[n:])

    return (c1, c2)

#-----------------

def create_candidate(nseg):
    ### Creates a randomly gnerated candidate made up of nseg numbers.
    ### restrict_sol can restrict solutions to be either positive or negative

    candidate=[]
    for n in range(nseg):
        if restrict_sol < 0:
            candidate.append(np.random.randint(-nmax,0))
        if restrict_sol > 0:
            candidate.append(np.random.randint(nmax))
        else:
            candidate.append(np.random.randint(-nmax,nmax))

    return candidate

#-----------------

def create_starting_population(nseg,npop):
    ### Creates an entire population of randomly generated candidates

    population=[]
    for k in range(npop):

        population.append(create_candidate(nseg))

    return population

#-----------------

def mutate(candidate,mutation_prob,mutation_amount):
    ### Every point in a candidate has a chance to mutate.
    ### Mutations are created by changing a point by a value defined by mutation_amount.

    for n in range(nseg):
        if np.random.random() < mutation_prob:
            num=candidate[n]
            noise = np.random.randint(-nmax*mutation_amount/2,nmax*mutation_amount/2)
            if abs(num + noise) < nmax:
                candidate[n]=num + noise

    return candidate



#%%
def main(iterations):

    #set seed if repeatable = True
    if repeatable:
        np.random.seed(0)

    #define x values initially - to be used later
    x=np.linspace(x0,xN,nseg)

    #create a random population
    population=create_starting_population(nseg,npop)
    for i in range(iterations):
        new_population = []

        #grade current population
        scores=score_population(population)
        best = population[np.argmin(scores)]
        y_best=convert(best)
        best_weight = np.min(scores)

        #graph best solution every incout generations
        if i%incout==0:
            graph(x,y_best,i+1,best_weight,x_exp,y_exp)


        #breed solutions into new population
        for j in range(ncouples):
            c1, c2 = breed(population[pick_parent(scores)], population[pick_parent(scores)])
            new_population = new_population + [c1, c2]

        #mutate new children
        for j in range(len(new_population)):
            new_population[j] = np.copy(mutate(new_population[j], mutation_prob, mutation_amount))

        # keep members of previous generation
        new_population += [population[np.argmin(scores)]]
        for j in range(1, nwinners):
            keeper = pick_parent(scores)
            new_population += [population[keeper]]

        # add new random members
        while len(new_population) < npop:
            new_population += [create_candidate(nseg)]

        #replace the old population with a real copy
        population = copy.deepcopy(new_population)

        #print final answer
        if i==iterations-1:
            scores=score_population(population)
            best = population[np.argmin(scores)]
            y_best=convert(best)
            best_weight = np.min(scores)
            graph(x,y_best,i+1,best_weight,x_exp,y_exp)
            input("Maximum iterations reached. Press ENTER to close...")

#%%
#Run the algorithm
main(total_iterations)
