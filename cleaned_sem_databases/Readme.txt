This folder containes an example of how datasets can be cleaned and used for reaserch. The datasets here contain only semantic properties with unified names and values in columns. They can be freele merged and compared with each other.

To ensure uniformity the following simplifications were made:
1. In the catalan database differences were reported with respect to some properties between declarative and subjunctive complements. The values of columns 'projection through negation' and  'neg-raising' were determined using the following strategy:
    i. If the predicate accepts only one type of complement the value is determined with respect to a complement of this type. 
    ii. If the predicate accepts both declarative and subjunctive complements and the value of the cell is the same for both, then this value is used. 
    iii. If the predicate accepts both declarative and subjunctive complements, and one of them have value "typically..." and the other "neither" or "non-neg-raising" or "typically..." then the latter value is used.
    iv. If the predicate accepts both declarative and subjunctive complements, and they have values different then in (iii) then the value is "undecided".

2. The English dataset has only 2 anti-rogarive predicates, the acceptability of interrogatives e.g. for the verb believe is justified by 'I get the intended meaning'. I excluded this dataset from the analysis. 

3. In the german database the predicates which do not embedd clauses were deleted (whisper and be certain)

4. In the mandarin database differences were reported with respect to some properties between negating the predicate with 'bu' and 'mei' negation. The values of columns 'projection through negation' and  'neg-raising' were determined using the following strategy:
    i. If the predicate can be negated only with one of the negations, the value is determined with respect to this negation. 
    ii. If the predicate can be negated with both negations and the value of the cell for 'bu' is used, since 'mei' brings eventuality readings (Citations).

5. In the spanish database (similar to catalan) differences were reported with respect to some properties between declarative and subjunctive complements. The values of columns 'projection through negation' and  'neg-raising' were determined using the following strategy:
    i. If the predicate accepts only one type of complement the value is determined with respect to a complement of this type. 
    ii. If the predicate accepts both declarative and subjunctive complements and the value of the cell is the same for both, then this value is used. 
    iii. If the predicate accepts both declarative and subjunctive complements, and one of them have value "typically..." and the other "neither" or "non-neg-raising" or "typically..." then the latter value is used.
    iv. If the predicate accepts both declarative and subjunctive complements, and they have values different then in (iii) then the value is "undecided".

6. Swedish database distinguishes for some properties between two readings of the verb 'think': "have a thought" and  "belief opinion". The latter is used, as it os closer to the readings of the other databses. 
    The believe version with pa was chosen, as it os closer to the readings of the other databses
    vara nyfiken (be courious) - the two senses were divided into two predicates.

7. In the turkish database the term 'slightly odd' was used it was replaced with 'neither'. For agree the second interpretatioon was chosen.

8. Q-to-P anti-veridical is given value 0 in Q-to-P veridicality.

9. typically Q-to-P and P-to-Q were treated as not distributive/veridical