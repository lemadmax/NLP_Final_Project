To run my program:
1. run: <<python hw6.py %-training %-test>>
	generate test-feature and training-feature two files.
2. Use OpenNLP package files
	run: <<java -cp .;maxent-3.0.0.jar;trove.jar MEtrain training.feature model.chunk>>
3. run: <<java -cp .;maxent-3.0.0.jar;trove.jar MEtag test.feature model.chunk response.chunk>> 
4. run: <<python gen_out.py response.chunk %-test>>
	generate partitive.txt: final result file
5. run: <<score.chunk.py %-test partitive.txt>>
	calculate accuracy and F1 score.

My final score: 
4127 out of 4276 tags correct
  accuracy: 96.52
400 groups in key
337 groups in response
294 correct groups
  precision: 87.24
  recall:    73.50
  F1:        79.78

I have tried 5 tags: POS, BIO, Previous_POS, Previous_BIO, and DisToPred(distance to the pred)
Before I added DisToPred, accuracy is about 88 with 65 F1 score. The result above is generated
after I added DisToPred.