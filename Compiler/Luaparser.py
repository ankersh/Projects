import re
import shlex
import sys

DO = "do"
WHILE = "while"
END = "end"
IF = "if"
ELSEIF = "elseif"
ELSE = "else"
UNTIL = "until"
FOR = "for"
REPEAT = "repeat"
THEN = "then"
FUNCTION = "function"
LOCAL = "local"
IN = "in"
END = "end"
RETURN = "return"
BREAK =  "break"
FALSE = "false"
TRUE = "true"
NIL = "nil"
terminals = [DO,WHILE,END,IF,ELSEIF,IN,ELSE,UNTIL,FOR,REPEAT,FUNCTION,LOCAL,END,RETURN,BREAK,FALSE,THEN,TRUE,NIL]                               #These are all the terminals which are words
binop = ['+','-','*','/','^','%','..','<','<=','>','>=','==','~=','and','or']                                                                   #These are everything that come under the Binop rule
unop = ['-','not','#']                                                                                                                          #These are everything that come under the Unop rule

Name =  re.compile("^[_a-zA-Z]\w*$")
Num = re.compile('^[0-9]+$')                                                    #Matches any number. I should have modified this to recognise numbers with decimal places and numbers in standard form but I didn't think about that until just now
String = re.compile("['\"]([^\"]*)['\"]")                                       #This essentially just matches anything between quotation marks, perfect for getting strings

start = 0

def StT(i):                         #This method is used to convert raw shlex tokens into tokens which are useful to my parser.


    global endCount                 #Keeps track of amount of "ends" in the program, even if our errorRecovery method misses one it will still tell the user theres one missing
    decimalPlaces = 2               #Used for counting how many decimal places a number has, if it has one
    if (i>=len(TempTokens)):        #Checks to see if we can access a token instead of causing an error
        return 'Error, out of range'
    cToken = TempTokens[i]
    if (cToken == END):
        endCount = endCount + 1
        return cToken
    if((i+1) < len(TempTokens)):
        if ((cToken == '=')|(cToken == '<')|(cToken == '>')|(cToken == '~')):                       #I realise shlex will naturally seperate binops if theyre two different symbols, so this checks to make sure thats not the case
            if (TempTokens[i+1] == '='):                                                            #and if it is, it will lump them together into one token
                del TempTokens[i+1]
                return 'BINOP'
        elif(cToken == '.'):
            if ((TempTokens[i+1] == '.')&((i+2) < len(TempTokens))):            #This did the same thing BUT we might have the nonterminal '...'. So we check for '...' before returning '..'
                if (TempTokens[i+2] == '.'):                                    #Unfortunately this COULD lead to an out of range error if the program ended in '..' for some reason
                    del TempTokens[i+1]                                         #so additional checks have to be carried out.
                    del TempTokens[i+1]
                    return '...'
                else:
                    del TempTokens[i+1]
                    return 'BINOP'
            else:
                if (TempTokens[i+1] == '.'):
                    del TempTokens[i+1]
                    return 'BINOP'
    if cToken == '-':                                   #This could be binop or unop, we don't know, so my rules check for (binop|'-') and (unop|'-') instead of binop and unop.
        return '-'
    if cToken == 'EOF':
        return 'EOF'
    for item in terminals:
        if (item == cToken):
            return cToken
    for item in binop:
        if (item == cToken):
            return 'BINOP'
    for item in unop:
        if (item == cToken):
            return 'UNOP'
    if Name.match(cToken):
        return 'NAME'
    elif Num.match(cToken):
        if((i+2) < len(TempTokens)):
            if (TempTokens[i+1] == '.'):
                if(Num.match(TempTokens[i+2])):                                     #If we find a decimal place after the number, put all the decimals after into the same NUM token
                    del TempTokens[i+1]
                    while(Num.match(TempTokens[i+1])):
                        del TempTokens[i+1]

        return 'NUM'
    elif String.match(cToken):
        return 'STRING'
    else: return cToken

def found(cToken):                              #This just checks to see if we find a token, if we do it moves on to the next token and returns true, if not the parser does nothing
    if (cToken == token):
        getToken()
        return True
    else:
        return False

errorTokenNo = 0                #This variable is used to track where abouts we found the error, used for error messages
expectedToken = ""              #This variable stores the token we expected to find if we have an error with check()


def check(cToken):      #This function is used when the program MUST have a terminal at a specific place. If the terminal is not a found, it indicates that the program does not parse.
    global error
    global errorTokenNo
    global expectedToken
    global token
    if token == 'ERROR':
        return True
    if found(cToken):
        return True
    else:
        print "Expected " + cToken + "; Token was " + token
        token = 'ERROR'
        expectedToken = cToken
        errorTokenNo = counter-1
        error = 1

        return False

def currentIs(cToken):                  #Just returns if the current token is the one we're looking for.
    if (cToken == token):
        return True
    return False

def getToken():                 #Gets a new token
    global token
    global counter
    if counter>=len(Tokens):
        print "Tokens used up"
        print "Final token was " + str(token)
        token = 'EOF'
    else:
        new = Tokens[counter]
        old = token
        token = token.replace(old,new)                          #Replaces the old token string with the new one
        counter = counter + 1


def startParse():
    getToken()
    block()


########################################################                                    #Beginning of my grammar rules
def block():

    print 'Block'
    chunk()

def chunk():
    if currentIs('ERROR'):                 #All grammar rules have this at the top, essentially if we find an error this just stops everything parsing as soon as possible.
        return False
    print 'Chunk'

    if (currentIs('NAME')|currentIs('(')|currentIs(DO)|currentIs(WHILE)|currentIs(REPEAT)|currentIs(IF)|currentIs(FOR)|currentIs(FUNCTION)|currentIs(LOCAL)):
        stat()              #If the current symbol is one of these we have a statement
        if found(';'):      #If we find ';' it means there is another statement, call chunk again
            chunk()         #to find it.
        elif (currentIs('NAME')|currentIs('(')|currentIs(DO)|currentIs(WHILE)|currentIs(REPEAT)|currentIs(IF)|currentIs(FOR)|currentIs(FUNCTION)|currentIs(LOCAL)):
            chunk()
    if (currentIs(RETURN)|currentIs(BREAK)):
        laststat()
        found(';')


def stat():                         #All the rules here follow the rules in the modifed grammar I use. If you'd like to see these rules, check grammar.pdf
    if currentIs('ERROR'):
        return False
    print 'Statement'
    global functionNo
    global error
    global endNeeded
    global token
    global counter
    global start
    start = counter-1           #Indicates where the statement starts, so when we print out our error message we can print out the beginning of the statement
    tempC = 0
    if (token == 'EOF'):
        return False
    if found(DO):
        endNeeded = endNeeded + 1  #Just indicates how many end tokens we need, we compare it to how many we have at the end.
        block()
        check(END)
    elif found(WHILE):
        endNeeded = endNeeded + 1
        exp()
        check(DO)
        block()
        check(END)
    elif found(REPEAT):
        endNeeded = endNeeded + 1
        block()
        check(UNTIL)
        exp()
    elif found(IF):
        endNeeded = endNeeded + 1
        exp()
        check(THEN)
        block()
        while found(ELSEIF):
            exp()
            check(THEN)
            block()
        if found(ELSE):
            block()
        check(END)
    elif found(FOR):
        endNeeded = endNeeded + 1
        namelist(0)                             #The reason namelist has an argument is to know if we need to store it as a functionName or not. In this case we don't
        if found('='):
            exp()
            check(',')
            exp()
            if found(','):
                exp()
            check(DO)
            block()
            check(END)
        elif found(IN):
            explist()
            check(DO)
            block()
            check(END)
    elif found(FUNCTION):
        endNeeded = endNeeded + 1
        functionNo = functionNo + 1         #Increases the number of functions we have found by 1, so we know where to store the parameters and name in the list storing them
        funcname()
        funcbody(1)                                 #Funcbody has a argument which it passes on to namelist and parlist
    elif found(LOCAL):
        if found(FUNCTION):                         #If we find local, there are two rules we can use. Fortunately we can easily pick between them by seeing if we have NAME next or FUNCTION
            endNeeded = endNeeded + 1
            check('NAME')
            functionNo = functionNo + 1
            print str(functionNo)
            functionNames.append(TempTokens[counter-2])
            funcbody(1)
        else:
            namelist(0)
            if found('='):
                print "just found ="
                explist()
    else:                                           #Our statement either is Varlist = Explist OR a functioncall. Unfortunately varlist might just be in the form of a functioncall
        if currentIs('NAME'):
            if ((Tokens[counter] == ':')|(Tokens[counter] == '(')|(Tokens[counter] == '.')):  #If this is true, we either have functioncall or varlist with a functioncall
                tempCounter = counter
                functioncall()
                if currentIs('=')|currentIs(','):                   #If this is true the functioncall is part of a varlist
                    print "This functioncall is part of a varlist"  #so backtrack and use the varlist rule
                    counter = tempCounter                           #nicely enough, even though we are backtracking there is no chance we accidentally trigger an error that shouldnt be there (i think)
                    token = Tokens[counter-1]                       #so we don't need to worry about that
                    varlist()
                    check('=')
                    explist()
            else:                                   #In this case we definitely have a Varlist
                varlist()
                check('=')
                explist()



def laststat():
    if currentIs('ERROR'):
        return False
    print 'LastStat'

    if found(RETURN):
        explist()
    else: check(BREAK)

def funcname():
    if currentIs('ERROR'):
        return False
    print 'Funcname'
    check('NAME')
    functionNames.append(TempTokens[counter-2])
    while found('.'):
        check('NAME')
        functionNames[functionNo] = functionNames[functionNo] + "." + TempTokens[counter-2]      #Adds the functionName to the list storing them
    if found(':'):
        check('NAME')
        functionNames[functionNo] = functionNames[functionNo] + ":" + TempTokens[counter-2]


def varlist():
    if currentIs('ERROR'):
        return False
    print 'Varlist'
    var()
    if found(','):                #If we find ',' there is another variable so run varlist again
        varlist()

def namelist(X):
    print "Namelist"
    if currentIs('ERROR'):
        return False
    if (X == 0):                     #If X = 0 we aren't dealing with the parameters of a function
        check('NAME')
        while found(','):
            if currentIs('...'):
                return True         #We just need to break at this point, its the end of namelist
            else:
                check('NAME')
    elif(X == 1):                       #If X = 1 we're dealing with parameters of a function so
        check('NAME')                      #save them to an array
        functionParameters.append(TempTokens[counter-2])
        while found(','):
            if found('...'):        #this is slightly wrong because namelist should hand this back to parlist to find, but the end result is exactly the same
                functionParameters[functionNo] = functionParameters[functionNo] + ", ..."
            else:
                check('NAME')
                print functionNo
                functionParameters[functionNo] = functionParameters[functionNo] + ", " + TempTokens[counter-2]

def explist():
    if currentIs('ERROR'):
        return False
    print 'Explist'

    exp()
    if found(','):              #If we find ',' we know there is another exp after so just call
        explist()               #explist again

def args():
    if currentIs('ERROR'):
        return False
    print 'Args'

    if found('STRING'):
        return True
    elif found('('):
        if found(')'):
            return True
        else:
            explist()
            check(')')
    else: tableconstructor()


def function():
    if currentIs('ERROR'):
        return False
    print 'Function'

    check(FUNCTION)
    funcbody(0)

def funcbody(X):
    if currentIs('ERROR'):
        return False
    print 'Funcbody'

    check('(')
    if found(')'):
        if(X == 1):              #If X = 1 this is a function being defined, not a function being called
            functionParameters.append("This function takes no parameters")
    else:
        parlist(X)
        check(')')
    block()
    check(END)

def parlist(X):
    if currentIs('ERROR'):
        return False
    print 'Parlist'
    if found('...'):
        if (X == 1):
            print str(functionNo)
            functionParameters.append("...")
    else:
        namelist(X)
        found(',')
        if found('...'):
            functionParameters.append(", ...")


def tableconstructor():
    if currentIs('ERROR'):
        return False
    print 'Tableconstructor'

    check('{')
    if found('}'):
        return True
    else:
        fieldlist()
        check('}')

def fieldlist():
    if currentIs('ERROR'):
        return False
    print 'Fieldlist'

    global counter
    global token
    global error

    field()                     #We need backtracking to get this to work
    while (found(',')|found(';')):
        if token != '}':          #The next token has to be the end of tableconstructor (so token = '{')
            field()               #or another field




def field():
    if currentIs('ERROR'):
        return False
    print 'Field'

    if found('['):
        exp()
        check(']')
        check('=')
        exp()
    elif currentIs('NAME'):
        if (Tokens[counter] == '='):    #We might have rule 2, 'Name '=' exp()'
            check('NAME')               #but at the same time we might have a functioncall or something
            check('=')                  #similar. So we see if rule 2 fits, if not we do exp()
            exp()
        else:
            exp()
    else:
        exp()

def var():
    if currentIs('ERROR'):
        return False
    print 'Var'
    if found('NAME'):
        var2()
    else:
        check('(')
        exp()
        check(')')
        prefixexp2()
        if found('['):
            exp()
            check(']')
            var2()
        elif found('.'):
            check('NAME')
            var2()

def var2():
    if currentIs('ERROR'):
        return False
    print 'Var2'
    if (currentIs(':')|currentIs('(')|currentIs('STRING')|currentIs('{')):      #If the current symbol is one of these we need to use prefixexp2 next because we either have Args next or rule 2 of prefixexp2
        prefixexp2()
        if found('['):
            exp()
            check(']')
            var2()
        elif found('.'):
            check('NAME')
            var2()
    elif found('['):            #These are the same as before, but this happens in the case where
        exp()                   #prefixexp2 does nothing. I think I could just run prefixexp2
        check(']')              #regardless though
        var2()
    elif found('.'):
        check('NAME')
        var2()




def prefixexp():
    if currentIs('ERROR'):
        return False
    print 'Prefixexp'
    global counter
    global token
    tempCounter = counter

    if found('('):
        exp()
        check(')')
        prefixexp2()
        if (currentIs('[')|currentIs('.')):    #If we actually should have used var function
            counter = tempCounter              #We will find one of these two symbols next
            token = Tokens[counter-1]          #so backtrack and use the other rule instead
            var()
            prefixexp2()
    else:
        var()
        prefixexp2()

def prefixexp2():
    if currentIs('ERROR'):
        return False
    print 'Prefixexp 2'
    global counter
    global error
    tempCounter = counter


    if found(':'):
        check('NAME')
        args()
        prefixexp2()
    elif(currentIs('(')|currentIs('STRING')|currentIs('{')):
        args()
        prefixexp2()



def functioncall():
    if currentIs('ERROR'):
        return False
    print 'Functioncall'
    prefixexp()
    if(currentIs('(')|currentIs('STRING')|currentIs('{')):
        args()

    elif found(':'):
        check('NAME')
        args()

def exp():
    print 'Exp'
    if currentIs('ERROR'):
        return False
    if found(NIL):
        exp2()
    elif found(FALSE):
        exp2()
    elif found(TRUE):
        exp2()
    elif found('NUM'):
        exp2()
    elif found('STRING'):
        exp2()
    elif found('...'):
        exp2()
    elif currentIs(FUNCTION):
        function()
        exp2()
    elif currentIs('{'):
        tableconstructor()
        exp2()
    elif found('UNOP')|found('-'):
        exp()
        exp2()
    else:
        prefixexp()
        exp2()

def exp2():
    if currentIs('ERROR'):
        return False
    print 'Exp2'

    if found('BINOP')|found('-'):
        exp()
        exp2()
############################################### End of grammar rules



def parse(filename):                                        #Uses shlex to convert the input file to tokens
                                                            #Uses StT to convert those tokens into tokens usable by my parser
    inputfile = open(filename,'rt').read()
    lexer = shlex.shlex(inputfile)
    for token in lexer:
        TempTokens.append(token)
    TempTokens.append('EOF')                                 #Adds EOF to the end so when we hit the last token it can't be used more than once
                                                            # (if we expected the last two tokens to be END and instead only the last one was, if we didn't have the EOF token the END would be used twice and parser would think no errors)
    for i in range(0,len(TempTokens)):
        Type = StT(i)
        if (Type != 'Error, out of range'):
            Tokens.append(Type)

    print Tokens                                    #Prints out a list of tokens at the beginning
    startParse()


def errorRecovery():                            #My errorRecovery function wasn't my first choice of method. After finding a error my program only can really check the structure of Statements.
    global counter                              #it'll sometimes find more than one though which is better than nothing
    global token
    global error
    print "ERRRR"
    print str(error)
    print str(counter)
    while((counter < len(Tokens)) & (error == 0)):
        token = Tokens[counter-1]
        if ((token == DO)|(token == WHILE)|(token == REPEAT)|(token == IF)|(token == FOR)|(token == FUNCTION)|(token == LOCAL)):                #We can easily tell we have a statement due to the terminals only found at the beginning of statements
            block()
        elif (token == 'NAME'):                                                                             #We might have the start of a new statement but we also might be in the middle of a varlist or parlist and we'll end up generating
            tempCounter = counter                                                                           #loads of errors that shouldn't really be there if we do this. Unfortunately we might miss out on a legitmately incorrect statement
            varlist()                                                                                       #if it happens to follow right after another one.
            check('=')
            explist()
            if(error == 1):
                error = 0
                counter = tempCounter
                token = Tokens[counter-1]
                functioncall()
                if(error ==1):
                    error = 0
                    counter = tempCounter + 1
                    token = Tokens[counter-1]
        else:
            counter = counter + 1

def errorMessage():                 #Just stores the error messages to be printed out after
    global errorMessages
    errorMessages.append("")
    errorMessages.append("Error with program, did not parse. Error found at token " + str(errorTokenNo) + "(" + TempTokens[errorTokenNo] + ")")
    errorstr = ""
    print ""
    for i in range(start,errorTokenNo+1):
        errorstr = errorstr + TempTokens[i] + " "
    errorMessages.append(errorstr)
    errorMessages.append( "expected " + expectedToken + ", instead found " + TempTokens[errorTokenNo])





functionNo = -1                 #Tracks the number of functions we've found
functionNames = []              #Stores the function names
functionParameters = []         #Stores the function parameters
TempTokens = []                 #Stores the raw tokens
Tokens = []                     #Stores the converted token
errorMessages = []              #Stores error messages


token = ""                      #Holds current token
counter = 0                     #Used to count what token we are on
error = 0                       #Used to indicate that an error has been found. error = 1 if error has been found. Will be changed back if we try to find additional errors
foundError = 0                  #Another indicator that an error has been found, once changed to 1 cannot be changed back.
endCount = 0                    #Counts amount of END tokens we found
endNeeded = 0                   #Counts amount of END tokens we need

print ""


if __name__ == "__main__":
    import sys
    parse(sys.argv[1])

if (error == 0):        #No errors found, the program parses, if there are any functions print then
    print ""
    print "No errors found"
    if (functionNo > -1):
        print "Functions found:"
        print ""
        for i in range(0, functionNo+1):
            print "function " + str(functionNames[i]) + "(" + functionParameters[i] + ")"
            print ""





if error == 1:      #An error has been found
    error = 0
    while (counter < len(Tokens)):          #Whilst we aren't yet on the last token, lets try to see if we can find an error in any more statements
        error = 0
        errorMessage()
        errorRecovery()
        counter = counter + 1
    if ((errorTokenNo+1) == len(Tokens)):                           #If we find an error RIGHT at the end of the program, we still need to add the error message
        errorMessage()
    print "Errors found"
    for i in errorMessages:
        print i

    if (endNeeded != endCount):         #Prints out how many ENDs are missing from the program
        print ""
        print ("The program is missing " + str(endNeeded-endCount) + " end tokens")
