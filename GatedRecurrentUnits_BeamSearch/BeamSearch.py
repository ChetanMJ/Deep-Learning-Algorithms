import numpy as np
import math

def GreedySearch(SymbolSets, y_probs):
    '''
    SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
    y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			your implementation for part 2 you need to incorporate batch_size.

    Return the forward probability of greedy path and corresponding compressed symbol 	  sequence i.e. without blanks and repeated symbols.
    '''
    max_index = np.argmax(y_probs, axis = 0)
    max_prob = np.max(y_probs, axis = 0)
    max_prob_path = np.prod(max_prob, axis = 0)
    SymbolSets.insert(0,' ')

    batch_size = y_probs.shape[2]
    Seq_length = y_probs.shape[1]

    compressed_symbols = []
    
    for i in range(batch_size):
        compressed_string = ''
        previous_symbol = ''
        for j in range(Seq_length):
            current_symbol = SymbolSets[max_index[j,i]]
            if (current_symbol != previous_symbol) and (current_symbol != ' ') :
                compressed_string = compressed_string + current_symbol
            previous_symbol = current_symbol
            
        compressed_symbols.append(compressed_string)
        
    return np.array(compressed_symbols), max_prob_path




def InitializePaths(SymbolSet, Y_Probs, BlankPathScore, PathScore, BeamWidth):
    path = ''
    BlankPathScore[''] = Y_Probs[0]
    InitialPathWithFinalBlank = {path}
    
    InitialPathWithFinalSymbol = set()
    
    
    for i in range(len(SymbolSet)):
        path = SymbolSet[i]
        PathScore[path] = Y_Probs[i+1]
        InitialPathWithFinalSymbol.add(path)
        
    return  Prune(InitialPathWithFinalBlank, InitialPathWithFinalSymbol, BlankPathScore, PathScore, BeamWidth)


def ExtendWithBlank(PathWithTerminalBlank, PathWithTerminalSymbol, Y_Probs, BlankPathScore, PathScore):
    UpdatedPathWithTerminalBlank = set()
    UpdatedBlankPathScore = {}
    
    for path in PathWithTerminalBlank:
        UpdatedPathWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * Y_Probs[0]
        
    for path in PathWithTerminalSymbol:
        if path in UpdatedPathWithTerminalBlank:
            UpdatedBlankPathScore[path] = UpdatedBlankPathScore[path] + (PathScore[path] * Y_Probs[0])
        else:
            UpdatedPathWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * Y_Probs[0]
            
    return UpdatedPathWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathWithTerminalBlank, PathWithTerminalSymbol,SymbolSet, Y_Probs, BlankPathScore, PathScore):
    UpdatedPathWithTerminalSymbol = set()
    UpdatedPathScore = {}
    
    
    for path in PathWithTerminalSymbol:
        for c in range(len(SymbolSet)):
            if SymbolSet[c] == path[-1]:
                newpath = path
            else:
                newpath = path + SymbolSet[c]
                
            if newpath in UpdatedPathWithTerminalSymbol:
                UpdatedPathScore[newpath] = UpdatedPathScore[newpath] + (PathScore[path] * Y_Probs[c+1])
            else:        
                UpdatedPathWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * Y_Probs[c+1]
            
    for path in PathWithTerminalBlank:
        for c in range(len(SymbolSet)):
            newpath = path + SymbolSet[c]
            
            if newpath in UpdatedPathWithTerminalSymbol:
                UpdatedPathScore[newpath] = UpdatedPathScore[newpath] + (BlankPathScore[path] * Y_Probs[c+1])
            else:
                UpdatedPathWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = BlankPathScore[path] * Y_Probs[c+1]

    return UpdatedPathWithTerminalSymbol, UpdatedPathScore


def Prune(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    PrunedPathWithTerminalBlank = set()
    PrunedPathWithTerminalSymbol = set()
    scorelist = []
    
    for p in PathWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
        
    for p in PathWithTerminalSymbol:
        scorelist.append(PathScore[p])
        
    scorelist.sort(reverse = True)
    cutoff = scorelist[BeamWidth]
    
    for p in PathWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]
            
    for p in PathWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
            
    return PrunedPathWithTerminalBlank, PrunedPathWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore
    

def MergeIdenticalPaths(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore):
    FinalPathScore = {}
    MergedPaths = PathWithTerminalSymbol
    for p in MergedPaths:
        FinalPathScore[p] = PathScore[p]
        
    for p in PathWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] = FinalPathScore[p] + BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]

    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    '''
    SymbolSets: This is the list containing all the symbols i.e. vocabulary (without 				  blank)
	
    y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your 			   batch size for part 1 would always remain 1, but if you plan to use 			your implementation for part 2 you need to incorporate batch_size.
	
    BeamWidth: Width of the beam.
	
    The function should return the symbol sequence with the best path score (forward 	  probability) and a dictionary of all the final merged paths with their scores. 
    '''
    ##print(y_probs)
    
    PathScore = {}
    BlankPathScore = {}

    PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore = InitializePaths(SymbolSets,y_probs[:,0,0], BlankPathScore, PathScore, BeamWidth)

    
    
    for i in range(1, y_probs.shape[1]):
        UpdatedPathWithTerminalBlank, UpdatedBlankPathScore = ExtendWithBlank(PathWithTerminalBlank, PathWithTerminalSymbol, y_probs[:,i,0], BlankPathScore, PathScore)
      
        UpdatedPathWithTerminalSymbol, UpdatedPathScore = ExtendWithSymbol(PathWithTerminalBlank, PathWithTerminalSymbol, SymbolSets, y_probs[:,i,0], BlankPathScore, PathScore)

        PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore  = \
        Prune(UpdatedPathWithTerminalBlank,UpdatedPathWithTerminalSymbol,UpdatedBlankPathScore,UpdatedPathScore,BeamWidth)

        

    MergedPaths, FinalPathScore = MergeIdenticalPaths(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore)

    BestPath = sorted(FinalPathScore, key=(lambda key:FinalPathScore[key]), reverse=True)
    
    print(FinalPathScore)
    print(BestPath[0])
   
    return BestPath[0], FinalPathScore
























'''
x = np.array([[[0.2, 0.1],
        [0.6, 0.4],
        [0.2, 0.7],
        [0.1, 0.2]],

       [[0.7, 0.8],
        [0.2, 0.5],
        [0.4, 0.1],
        [0.8, 0.1]],

       [[0.1, 0.1],
        [0.2, 0.1],
        [0.5, 0.2],
        [0.1, 0.7]]])
    
symbolset = ['a','b']

strs, prob= GreedySearch(symbolset, x)

print(strs)
print(prob)

x = np.array([[[0.2, 0.1],
        [0.6, 0.4],
        [0.2, 0.7],
        [0.1, 0.2]],

       [[0.7, 0.8],
        [0.2, 0.5],
        [0.4, 0.1],
        [0.8, 0.1]],

       [[0.1, 0.1],
        [0.2, 0.1],
        [0.5, 0.2],
        [0.1, 0.7]]])
    
symbolset = ['a','b']
seqs, probs = BeamSearch(symbolset, x, 2)

print("Beam search result")
print(seqs)
print(probs)
'''

#y_rands = np.random.uniform(0.001, 1.0, (4,10,1))
#y_sum = np.sum(y_rands, axis=0)
#y_probs = y_rands/y_sum
'''
y_probs = np.array([[[ 0.26405534],
  [ 0.08455321],
  [ 0.11432756],
  [ 0.24741288],
  [ 0.04386061],
  [ 0.36503554],
  [ 0.33549824],
  [ 0.1985073 ],
  [ 0.11844802],
  [ 0.07018962]],

 [[ 0.28135575],
  [ 0.34695366],
  [ 0.21286159],
  [ 0.07708888],
  [ 0.01402863],
  [ 0.33716491],
  [ 0.31626539],
  [ 0.17253898],
  [ 0.09482154],
  [ 0.69640945]],

 [[ 0.02822815],
  [ 0.23596769],
  [ 0.22783893],
  [ 0.31031899],
  [ 0.34444777],
  [ 0.1944168 ],
  [ 0.11827125],
  [ 0.30634359],
  [ 0.67797919],
  [ 0.00191067]],

 [[ 0.42636076],
  [ 0.33252544],
  [ 0.44497192],
  [ 0.36517925],
  [ 0.59766299],
  [ 0.10338275],
  [ 0.22996513],
  [ 0.32261013],
  [ 0.10875125],
  [ 0.23149026]]])

SymbolSets = ['a','b','c']
BeamWidth = 3
#print(y_probs)
a,b = BeamSearch(SymbolSets, y_probs, BeamWidth)
print(a,b)

test#2

np.random.seed(4)
EPS = 1e-5
#print(EPS)
y_rands = np.random.uniform(EPS, 1.0,(5,20,1))
y_sum = np.sum(y_rands, axis = 0)
y_probs = y_rands/y_sum

Symbolsets = ['a','b','c']

beamwidth = 3
a,b = BeamSearch(Symbolsets,y_probs,beamwidth )


'''