/*
 * Copyright 2011 Arnim Bleier, Andreas Niekler and Patrick Jaehnichen
 * Licensed under the GNU Lesser General Public License.
 * http://www.gnu.org/licenses/lgpl.html
 */

package edu.drexel.sentiment;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

/**
 * Hierarchical Dirichlet Processes  
 * Chinese Restaurant Franchise Sampler
 * 
 * 
 */
public class HDPRatingTester {
    HDPModel trainModel = null;
    HDPModel testModel = null;

	private double[] p;
	private double[] f;
	private double[][][] f_ksu;
	
	private int K;     // dynamic variable  >= trainModel.K
	private int T;     // dynamic variable  >= trainModel.T

	/**
	 * Initially assign the words to tables and topics
	 * 
	 * @param corpus {@link Corpus} on which to fit the model
	 */
	public void setModel(HDPModel trainModel, HDPModel testModel) {
		this.trainModel = trainModel;
		this.testModel = testModel;

		p = new double[testModel.K + 1]; 
		f = new double[testModel.K + 1];
		f_ksu = new double[testModel.K + 1][testModel.S][testModel.U];
	}

	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextGibbsSweep() {
		int t, k;
		for (int d = 0; d < testModel.D; d++) {
			DOCState docState = testModel.docStates[d];
			for (int i = 0; i < docState.documentLength; i++) {
				testModel.removeWord(d, i);    // remove the word i from the state
				computeF(d, i);      // f will be evaluated here
				t = sampleTable(d, i);
				if (t == docState.numberOfTables) {  // new Table
					// sampling its Topic and other latent variables
					k = sampleTopic(); 
				} else {
					// existing Table
					k = docState.tableToTopic[t];
				}
				
				LatentVariables var = sampleOtherLatentVariables(k);
				testModel.addWord(d, i, t, k, var.s, var.u);
			}
		}
		testModel.defragment(trainModel);
	}

	protected double[][][] computeF(int d, int i) {
		int k, s, u;
		double vb = testModel.V * testModel.beta;
		double slamda = testModel.S * testModel.lamda;
		double ueta = testModel.U * testModel.eta;

		K = Math.max(testModel.K, trainModel.K);
		T = testModel.T + trainModel.T;
		
		DOCState docState = testModel.docStates[d];
		f = Utils.ensureCapacity(f, K);
		f_ksu = Utils.ensureCapacity(f_ksu, K);

		int x = docState.author;
		int w = docState.words[i].w;
		
		// compute f_ksu   0~trainModel.K-1
		if (x >= trainModel.X) {  // the user not exists in train dataset
			for (k = 0; k < trainModel.K; k++) {
				f[k] = 0; // initial
				for (s = 0; s < trainModel.S; s++) {
					for (u = 0; u < trainModel.U; u++) {
						f_ksu[k][s][u] = (testModel.c_kxu[k][x][u] + testModel.eta)
								/ (testModel.c_kxu[k][x][0] + testModel.c_kxu[k][x][1] + ueta)
								* (trainModel.l_kw[k][w] + testModel.l_kw[k][w] + testModel.beta)
								/ (trainModel.l_k[k] + testModel.l_k[k] + vb)
								* (trainModel.l_kws[k][w][s]
										+ testModel.l_kws[k][w][s] + testModel.lamda)
								/ (trainModel.l_kw[k][w] + testModel.l_kw[k][w] + slamda);
	
						f[k] += f_ksu[k][s][u];
					}
				}
			}
		} else {
			for (k = 0; k < trainModel.K; k++) {
				f[k] = 0; // initial
				for (s = 0; s < trainModel.S; s++) {
					for (u = 0; u < trainModel.U; u++) {
						f_ksu[k][s][u] = (trainModel.c_kxu[k][x][u] + testModel.c_kxu[k][x][u] + testModel.eta)
								/ (trainModel.c_kxu[k][x][0] + trainModel.c_kxu[k][x][1] + testModel.c_kxu[k][x][0] + testModel.c_kxu[k][x][1] + ueta)
								* (trainModel.l_kw[k][w] + testModel.l_kw[k][w] + testModel.beta)
								/ (trainModel.l_k[k] + testModel.l_k[k] + vb)
								* (trainModel.l_kws[k][w][s]
										+ testModel.l_kws[k][w][s] + testModel.lamda)
								/ (trainModel.l_kw[k][w] + testModel.l_kw[k][w] + slamda);
	
						f[k] += f_ksu[k][s][u];
					}
				}
			}
		}

		// compute f_ksu trainModel.K~testModel.K-1
		for (; k < testModel.K; k++) {
			f[k] = 0;  //initial 
			for (s = 0; s < testModel.S; s++) {
				for (u = 0; u < testModel.U; u++) {
					f_ksu[k][s][u] = (testModel.c_kxu[k][x][u] + testModel.eta) / (testModel.c_kxu[k][x][0] + testModel.c_kxu[k][x][1] + ueta)
							* (testModel.l_kw[k][w] + testModel.beta) / (testModel.l_k[k] + vb)
							* (testModel.l_kws[k][w][s] + testModel.lamda) / (testModel.l_kw[k][w] + slamda);
					
					f[k] += f_ksu[k][s][u];
				}
			}
		}

		// f[K] is f_new
		f[k] = 0;
		for (s = 0; s < testModel.S; s++) {
			for (u = 0; u < testModel.U; u++) {
	
		        f_ksu[k][s][u] = 1.0 / testModel.U / testModel.V / testModel.S;
		        f[testModel.K] += f_ksu[testModel.K][s][u];
			}
		}
		//System.out.println(String.format("l_ds[%d](%2d,%2d,%2d)/%d, F[K]:%4.6f", d, l_ds[d][0], l_ds[d][1], l_ds[d][2], N, f[K]));

		return f_ksu;
	}

	private int sampleTopic() {
		double q, pSum = 0.0;
		int k = 0;

		p = Utils.ensureCapacity(p, K);
		for (k = 0; k < trainModel.K; k++) {
			pSum += (trainModel.m_k[k] + testModel.m_k[k]) * f[k];
			p[k] = pSum;		
		}

		for (; k < testModel.K; k++) {
			pSum += testModel.m_k[k] * f[k];
			p[k] = pSum;		
		}

		pSum += testModel.gamma * f[k];
		p[k] = pSum;
		
		q = testModel.random.nextDouble() * pSum;
		for (k = 0; k <= K; k ++) {
			if (q < p[k])
				break;
		}

		return k;
	}

	/**
	 * Decide other latent variables
	 * 
	 * @return the indexes of the s and u
	 */
	private LatentVariables sampleOtherLatentVariables(int k) {
		LatentVariables vars = new LatentVariables();

		double q, pSum = 0.0;
		int s, u;
		p = Utils.ensureCapacity(p, testModel.S * testModel.U);
		
		int index = 0;

		// for existing k
		for (s = 0; s < testModel.S; s++) {
			for (u = 0; u < testModel.U; u++) {
				pSum += f_ksu[k][s][u];

				p[index++] = pSum;
			}
		}

		q = testModel.random.nextDouble() * pSum;
		for (index = 0; index < testModel.S * testModel.U; index ++) {
			if (q < p[index])
				break;
		}
		// index
		vars.s = index / testModel.U % testModel.S;
		vars.u = index % testModel.U;

		return vars;
	}

	/**	 
	 * Decide at which table the word should be assigned to
	 * 
	 * @param docID the index of the document of the current word
	 * @param i the index of the current word
	 * @param f the f(w,u,s|z) probability
	 * @return the index of the table
	 */
	int sampleTable(int docID, int i) {
		int k, j;
		double pSum = 0.0, fNew, q;
		
		DOCState docState = testModel.docStates[docID];
		p = Utils.ensureCapacity(p, docState.numberOfTables);
		fNew = testModel.gamma * f[K];                //  if k = k_new

		// p(x)
		for (k = 0; k < trainModel.K; k++) {
			fNew += (trainModel.m_k[k] + testModel.m_k[k]) * f[k];
		}

		for (; k < testModel.K; k++) {
			fNew += testModel.m_k[k] * f[k];
		}

		// ready to sample table
		for (j = 0; j < docState.numberOfTables; j++) {
			if (docState.wordCountByTable[j] > 0) 
				pSum += docState.wordCountByTable[j] * f[docState.tableToTopic[j]];
			p[j] = pSum;
		}
		pSum += testModel.alpha * fNew / (T + testModel.gamma); // Probability for t = tNew
		p[docState.numberOfTables] = pSum;
		q = testModel.random.nextDouble() * pSum;
		for (j = 0; j <= docState.numberOfTables; j++)
			if (q < p[j]) 
				break;	// decided which table the word i is assigned to

		return j;
	}

	/**
	 * Method to call for fitting the model.
	 * 
	 * @param doShuffle
	 * @param shuffleLag
	 * @param maxIter number of iterations to run
	 * @param saveLag save interval 
	 * @param wordAssignmentsWriter {@link WordAssignmentsWriter}
	 * @param topicsWriter {@link TopicsWriter}
	 * @throws IOException 
	 */
	public void run(int shuffleLag, int maxIter, PrintStream log) 
	throws IOException {
		for (int iter = 0; iter < maxIter; iter++) {
			if ((shuffleLag > 0) && (iter > 0) && (iter % shuffleLag == 0))
				testModel.doShuffle();
			nextGibbsSweep();
			log.println(String.format("iter = %d, #topics = (Train:%d, Test:%d), #tables = (Train:%d, Test:%d)", iter, trainModel.K, testModel.K, trainModel.T, testModel.T));
		}
	}

	protected void printF() {
		double sum = 0, ssum = 0;
		for (int k = 0; k < K; k ++) {
			sum = 0;
			for (int s = 0; s < testModel.S; s ++) {
				for (int u = 0; u < testModel.U; u ++) {
					sum += testModel.m_k[k]*f_ksu[k][s][u];
					System.out.print(String.format("%1.5f\t", f_ksu[k][s][u]));
				}
			}
			ssum += sum;
			System.out.println(String.format("%1.5f = %1.5f(%2d)", sum, testModel.m_k[k]*f[k], testModel.m_k[k]));
		}
		
		sum = 0;
		for (int s = 0; s < testModel.S; s ++) {
			for (int u = 0; u < testModel.U; u ++) {
				sum += testModel.gamma*f_ksu[testModel.K][s][u];
				System.out.print(String.format("%1.5f\t", f_ksu[testModel.K][s][u]));
			}
		}
		ssum += sum;
		System.out.println(String.format("%1.5f = %1.5f(%3f)", sum, testModel.gamma*f[testModel.K], testModel.gamma));
		System.out.println(ssum);
	}

	/**
	 * The base of log should be 2 according to the original theory
	 * Now we set the base to e according to the paper of Blei 2003
	 * 
	 * For test corpus, the perplexity cannot be estimated 
	 * when the testModel.K > trainModel.K
	 * 
	 * @return
	 */
	public double computePerplexity() {
		double perp = 0.0;

		int N = 0, d, n, w, k;
		
		trainModel.estimateThetaPhi();
		testModel.estimateThetaPhi();
		
		for (d = 0; d < testModel.D; d ++) {
			DOCState docState = testModel.docStates[d];
			for (n = 0; n < docState.words.length; n ++) {
				w = docState.words[n].w;

				// log(p(w1)) + log(p(w2)) + ... 
				// Sigma^k( p(w|z=k) * (p(z=k) )
				double pw = 0.0;
				for (k = 0; k < trainModel.K; k ++) {
					pw += trainModel.phi[k][w] * testModel.theta[k][d];
				}
				for (; k < testModel.K; k ++) {
					pw += testModel.phi[k][w] * testModel.theta[k][d];
				}

				perp += Math.log(pw);
			}
			N += testModel.docStates[d].documentLength;
		}
		
		perp /= N;   //   log(p(w)) / N
		perp = Math.exp(-perp);

		return perp;
	}

	/**
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {
		/*HDPConfig trainConfig = new HDPConfig();
		trainConfig.loadConfig("./data/config.txt");
		
		Dictionary wDict = new Dictionary();
		Dictionary aDict = new Dictionary();
		
		// load training dictionaries
		wDict.loadDictionary(trainConfig.dir, "v.txt");
		aDict.loadDictionary(trainConfig.dir, "a.txt");

		HDPModel trainModel = new HDPModel();
		trainModel.loadModel(trainConfig);
		
		System.out.println(trainModel.D);
		System.out.println(trainModel.W);
		System.out.println(trainModel.V);
		System.out.println(trainModel.X);
		System.out.println(trainModel.S);
		System.out.println(trainModel.U);
		System.out.println(trainModel.K);
		System.out.println(trainModel.T);
		
		HDPConfig tempConfig = new HDPConfig();
		tempConfig.loadConfig("./temp/config.txt");
		trainModel.saveModel(tempConfig, wDict, aDict);*/

		// Configuration
		HDPConfig trainConfig = new HDPConfig();
		trainConfig.loadConfig(args[0]);

		HDPConfig testConfig = new HDPConfig();
		testConfig.loadConfig(args[1]);

		// Train Dictionaries
		Dictionary wDict = new Dictionary();
		Dictionary aDict = new Dictionary();
		Dictionary iDict = new Dictionary();
		
		// load training dictionaries
		wDict.loadDictionary(trainConfig.dir, "dict-v.txt");
		aDict.loadDictionary(trainConfig.dir, "dict-a.txt");
		iDict.loadDictionary(trainConfig.dir, "dict-i.txt");

		// Test Dataset
		Corpus corpus = new Corpus(wDict, aDict, iDict);
		corpus.readData(new File(testConfig.dir, testConfig.tData), new File(testConfig.dir, testConfig.sData), testConfig.dataCharset, true);

		// save Dictionaries
		wDict.saveDictionary(testConfig.dir, "dict-v.txt");
		aDict.saveDictionary(testConfig.dir, "dict-a.txt");
		iDict.saveDictionary(testConfig.dir, "dict-i.txt");

		// Load Training Model
		HDPModel trainModel = new HDPModel();
		trainModel.loadModel(trainConfig);
		
		// Create Test Model
		HDPModel testModel = new HDPModel();
		testModel.setConfig(testConfig);
		testModel.K = trainModel.K;
		testModel.addInstances(corpus);
		
		// Training
		HDPRatingTester hdp = new HDPRatingTester();
		hdp.setModel(trainModel, testModel);

		System.out.println("sizeOfVocabulary = " + testModel.V);
		System.out.println("totalNumberOfWords = " + testModel.W);
		System.out.println("NumberOfDocs = " + testModel.D);

		hdp.run(testConfig.shuffleLag, testConfig.niter, System.out);

		System.out.println(hdp.computePerplexity());

		// Save Model
		testModel.saveModel(testConfig, wDict, aDict);
	}
		
}