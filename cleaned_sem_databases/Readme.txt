This folder containes an example of how datasets can be cleaned and used for reaserch. The datasets here contain only semantic properties with unified names and values in columns. They can be freele merged and compared with each other.

To ensure uniformity the following simplifications were made:
1. In the catalan database differences were reported with respect to some properties between declarative and subjunctive complements. The values of columns 'projection through negation' and  'neg-raising' were determined using the following strategy:
    i. If the predicate accepts only one type of complement the value is determined with respect to a complement of this type. 
    ii. If the predicate accepts both declarative and subjunctive complements and the value of the cell is the same for both, then this value is used. 
    iii. If the predicate accepts both declarative and subjunctive complements, and one of them have value "typically..." and the other "neither" or "non-neg-raising" or "typically..." then the latter values is used.
    iv. If the predicate accepts both declarative and subjunctive complements, and they have values different then in (iii) then the value is "undecided".

2. The English dataset has only 2 anti-rogarive predicates, the acceptability of interrogatives e.g. for the verb believe is justified by 'I get the intended meaning'. I excluded this dataset from the analysis. 

3. In the german database the predicates which do not embedd clauses were deleted (whisper and be certain)