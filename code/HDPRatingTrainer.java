/*
 * Copyright 2011 Arnim Bleier, Andreas Niekler and Patrick Jaehnichen
 * Licensed under the GNU Lesser General Public License.
 * http://www.gnu.org/licenses/lgpl.html
 */

package edu.drexel.sentiment;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Formatter;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Hierarchical Dirichlet Processes  
 * Chinese Restaurant Franchise Sampler
 * 
 */
public class HDPRatingTrainer {
    HDPModel model = null;
    HDPConfig config = null;

	// variables repeatedly used for sampling 
    private double[] p;
	private double[] f;
	private double[][][] f_ksu;
	private LatentVariables vars = new LatentVariables();
	private double[][] tf;
	private double[][][][] tf_ksu;
	private int[]          tWords;        // words in the same table

	/**
	 * Initially assign the words to tables and topics
	 * 
	 * @param corpus {@link Corpus} on which to fit the model
	 */
	public void setModel(HDPModel model) {
		this.model = model;

		// max # of topics ~ 20
		p = new double[20]; 
		f = new double[20];
		f_ksu = new double[20][model.S][model.U];
		
		// max # of tables in each document  ~ 30
		tf = new double[30][20];
		tf_ksu = new double[30][20][model.S][model.U];
		
		// max # of words in each table ~ 100
		tWords = new int[1000];
	}

	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextGibbsSweep() {
		int t, k;
		for (int d = 0; d < model.D; d++) {
			DOCState docState = model.docStates[d];
			for (int i = 0; i < docState.documentLength; i++) {
				model.removeWord(d, i);    // remove the word i from the state
				computeF(d, i);            // f will be evaluated here
				t = sampleTable(d, i);
				if (t == docState.numberOfTables) {  // new Table
					// sampling its Topic and other latent variables
					k = sampleTopic();
					vars = sampleOtherLatentVariables(k);
					model.addWord(d, i, t, k, vars.s, vars.u); 
				} else {
					// existing Table
					k = docState.tableToTopic[t];
					LatentVariables var = sampleOtherLatentVariables(k);
					model.addWord(d, i, t, k, var.s, var.u);
				}
			}
		}
		model.defragment();
	}

	protected double[][][] computeF(int d, int i) {
		int k, s, u;
		double vb = model.V * model.beta;
		DOCState docState = model.docStates[d];
		f = Utils.ensureCapacity(f, model.K);
		p = Utils.ensureCapacity(p, docState.numberOfTables);
		f_ksu = Utils.ensureCapacity(f_ksu, model.K);

		int x = docState.author;
		int r = docState.rating;
		int w = docState.words[i].w;

		double r_gap, r_w;

		// compute f_ksu
		for (k = 0; k < model.K; k++) {
			f[k] = 0;  //initial 
			for (s = 0; s < model.S; s++) {
				for (u = 0; u < model.U; u++) {
					r_w = WordState.getR(s, u, model.config.neutRating);
					if (r_w != model.config.neutRating) {
						r_gap = r - (docState.r_sum + r_w)  / (docState.r_num + 1);
					} else if (docState.r_num == 0) {
						r_gap = r - model.config.neutRating;
					} else {
					    r_gap = r - docState.r_sum / docState.r_num;
					}
					double r_likelihood = Math.exp((-r_gap * r_gap) / 2 / model.sigma);

					f_ksu[k][s][u] = (model.c_kxu[k][x][u] + model.eta) / (model.c_kxu[k][x][0] + model.c_kxu[k][x][1] + 1)
							* (model.l_kw[k][w] + model.beta) / (model.l_k[k] + vb)
							* (model.l_kws[k][w][s] + model.lamda) / (model.l_kw[k][w] + model.S * model.lamda)
							* r_likelihood;

					f[k] += f_ksu[k][s][u];
				}
			}
		}

		// f[K] is f_new
		f[k] = 0;
		for (s = 0; s < model.S; s++) {
			for (u = 0; u < model.U; u++) {
				r_w = WordState.getR(s, u, model.config.neutRating);
				if (r_w != model.config.neutRating) {
					r_gap = r - (docState.r_sum + r_w)  / (docState.r_num + 1);
				} else if (docState.r_num == 0) {
					r_gap = r - model.config.neutRating;
				} else {
				    r_gap = r - docState.r_sum / docState.r_num;
				}

				double r_likelihood = Math.exp((-r_gap * r_gap) / 2 / model.sigma);

		        f_ksu[k][s][u] = 1.0 / model.U / model.V / model.S * r_likelihood;
		        f[model.K] += f_ksu[model.K][s][u];
			}
		}
		//System.out.println(String.format("l_ds[%d](%2d,%2d,%2d)/%d, F[K]:%4.6f", d, l_ds[d][0], l_ds[d][1], l_ds[d][2], N, f[K]));

		return f_ksu;
	}
	
	/**
	 * 
	 * @param d
	 * @param t
	 * @return
	 */
	private int sampleTopicForTable(int d, int t) {
		DOCState docState = model.docStates[d];
		tWords = Utils.ensureCapacity(tWords, docState.documentLength);
		int tCount = 0;
		int index = 0;

		// 找到同桌的所有的words excluding cur i
		for (int n = 0; n < docState.documentLength; n ++) {
			WordState ws = docState.words[n];
			if (ws.t == t) {      // 当前桌子上的单词
				tWords[tCount ++] = n;
			}
		}

		//1. remove words for other words
		for (index = 0; index < tCount; index ++) {
			model.removeWord(d, tWords[index]);
		}

		//2. compute f(k)  tWords.size()
		tf = Utils.ensureCapacity(tf, tCount);
		for (index = 0; index < tf.length; index ++) {
			tf[index] = Utils.ensureCapacity(tf[index], model.K);
		}
		tf_ksu = Utils.ensureCapacity(tf_ksu, tCount);
		for (index = 0; index < tf_ksu.length; index ++) {
			tf_ksu[index] = Utils.ensureCapacity(tf_ksu[index], model.K);
		}

		// Compute f_ksu for words in the same table
		for (index = 0; index < tCount; index ++) {
			computeF(d, tWords[index]);
			System.arraycopy(f, 0, tf[index], 0, model.K + 1);
			Utils.copy(f_ksu, tf_ksu[index]);
		}

		// 3. sample k
		p = Utils.ensureCapacity(p, model.K);
		double q, pSum = 0.0, fix = Math.log(10)*tCount;
		int k = 0;
		// if k is previously used
		//System.out.println("-----------------------------");
		for (k = 0; k < model.K; k ++) {
			q = Math.log(model.m_k[k]) + fix;
			for (index = 0; index < tCount; index ++) {
				q += Math.log(tf[index][k]);
			}
			q = Math.exp(q);
			//System.out.println(q + "\t" + model.m_k[k]);
			pSum += q;
			p[k] = pSum;
		}
		// if k = k^new
		q = Math.log(model.gamma) + fix;
		for (index = 0; index < tCount; index ++) {
			q += Math.log(tf[index][k]);
		}
		q = Math.exp(q);
		//System.out.println(q);
		pSum += q;
		p[k] = pSum;
		
		// sample k   
		// note: the probability of psum = 0 is very high
		q = model.random.nextDouble() * pSum;
		for (k = 0; k < model.K; k ++) {
			if (q < p[k])
				break;
		}

		//4. sample s and u and add all words including cur i
		for (index = 0; index < tCount; index ++) {
		    vars = sampleOtherLatentVariables(tf_ksu[index], k);
		    model.addWord(d, tWords[index], t, k, vars.s, vars.u);
		}

		return tCount;
	}

	/**
	 * 对已经存在的table重新sample一个topic
	 * 对概率算来说，这个步骤对最终结果影响不大
	 * 
	 * 由于 对不同的 word来说，f值相差太大，导致无法计算  
	 * 
	 * 算法：  
	 *   1 先将该table上所有的word remove掉
	 *   2 再计算每个word的f(k)
	 *   3 计算sampling概率并sample k
	 *   4 重新sample s和u
	 *   5 add words
	 *   
	 * 疑问： how to sample s and u in such circumstance
	 * 
	 * @param t
	 * @return
	 */
	private int sampleForTable(int d, int i, int t) {
		DOCState docState = model.docStates[d];
		tWords = Utils.ensureCapacity(tWords, docState.documentLength);
		int tCount = 0;
		int index = 0;

		// 找到同桌的所有的words excluding cur i
		for (int n = 0; n < docState.documentLength; n ++) {
			WordState ws = docState.words[n];
			if (ws.t == t && n != i) {      // 当前桌子上的单词
				tWords[tCount ++] = n;
			}
		}

		//1. remove words for other words
		for (index = 0; index < tCount; index ++) {
			model.removeWord(d, tWords[index]);
		}

		//2. compute f(k)  cur i + tWords.size()
		tf = Utils.ensureCapacity(tf, tCount);
		for (index = 0; index < tf.length; index ++) {
			tf[index] = Utils.ensureCapacity(tf[index], model.K);
		}
		tf_ksu = Utils.ensureCapacity(tf_ksu, tCount);
		for (index = 0; index < tf_ksu.length; index ++) {
			tf_ksu[index] = Utils.ensureCapacity(tf_ksu[index], model.K);
		}

		// Compute f_ksu for other words in the same table
		for (index = 0; index < tCount; index ++) {
			computeF(d, tWords[index]);
			System.arraycopy(f, 0, tf[index], 0, model.K + 1);
			Utils.copy(f_ksu, tf_ksu[index]);
		}
		// Compute f_ksu for current i in doc d
		computeF(d, i);
		System.arraycopy(f, 0, tf[tCount], 0, model.K + 1);
		Utils.copy(f_ksu, tf_ksu[tCount]);

		// 3. sample k
		p = Utils.ensureCapacity(p, model.K);
		double q, pSum = 0.0, fix = Math.log(10)*2*tCount;
		int k = 0;
		// if k is previously used
		for (k = 0; k < model.K; k ++) {
			q = Math.log(model.m_k[k])+fix;
			for (index = 0; index < tCount + 1; index ++) {
				q += Math.log(tf[index][k]);
			}
			q = Math.exp(q);
			pSum += q;
			p[k] = pSum;
		}
		// if k = k^new
		q = Math.log(model.gamma)+fix;
		for (index = 0; index < tCount + 1; index ++) {
			q += Math.log(tf[index][k]);
		}
		q = Math.exp(q);
		pSum += q;
		p[k] = pSum;
		
		// sample k   
		// note: the probability of psum = 0 is very high
		q = model.random.nextDouble() * pSum;
		for (k = 0; k < model.K; k ++) {
			if (q < p[k])
				break;
		}
		//System.out.println(model.K + "\t" + k + " " + pSum);

		//4. sample s and u and add all words including cur i
		for (index = 0; index < tCount; index ++) {
		    vars = sampleOtherLatentVariables(tf_ksu[index], k);
		    model.addWord(d, tWords[index], t, k, vars.s, vars.u);
		}
		// for cur 
		vars = sampleOtherLatentVariables(tf_ksu[index], k);
	    model.addWord(d, i, t, k, vars.s, vars.u);

		return 0;
	}

	private int sampleTopic() {
		double q, pSum = 0.0;
		int k = 0;

		p = Utils.ensureCapacity(p, model.K);
		for (k = 0; k < model.K; k++) {
			pSum += model.m_k[k] * f[k];
			p[k] = pSum;		
		}
		pSum += model.gamma * f[k];
		p[k] = pSum;

		q = model.random.nextDouble() * pSum;
		for (k = 0; k <= model.K; k ++) {
			if (q < p[k])
				break;
		}

		return k;
	}

	/**
	 * Decide at which topic the table should be assigned to
	 * 
	 * @return the index of the topic
	 */
	private LatentVariables sampleLatentVariables() {
		double q, pSum = 0.0;
		int k, s, u;
		p = Utils.ensureCapacity(p, (model.K + 1) * model.S * model.U);

		int index = 0;

		// for existing k
		for (k = 0; k < model.K; k++) {
			for (s = 0; s < model.S; s ++) {
				for (u = 0; u < model.U; u ++) {
					pSum += model.m_k[k] * f_ksu[k][s][u];

					p[index ++] = pSum;
				}
			}
		}

		// for k_new
		for (s = 0; s < model.S; s ++) {
			for (u = 0; u < model.U; u ++) {
				pSum += model.gamma * f_ksu[k][s][u];

				p[index ++] = pSum;
			}
		}

		q = model.random.nextDouble() * pSum;
		for (index = 0; index < (model.K + 1) * model.S * model.U; index ++) {
			if (q < p[index])
				break;
		}
		// index
		vars.k = index / model.U / model.S;
		vars.s = index / model.U % model.S;
		vars.u = index % model.U;

		return vars;
	}
	
	/**
	 * Decide other latent variables
	 * 
	 * @return the index of the topic
	 */
	private LatentVariables sampleOtherLatentVariables(int k) {
		return sampleOtherLatentVariables(f_ksu, k);
	}
	
	/**
	 * Decide other latent variables
	 * 
	 * @return the index of the topic
	 */
	private LatentVariables sampleOtherLatentVariables(double[][][] ksu, int k) {
		double q, pSum = 0.0;
		int s, u;
		p = Utils.ensureCapacity(p, model.S * model.U);
		
		int index = 0;

		// for existing k
		for (s = 0; s < model.S; s++) {
			for (u = 0; u < model.U; u++) {
				pSum += f_ksu[k][s][u];

				p[index++] = pSum;
			}
		}

		q = model.random.nextDouble() * pSum;
		for (index = 0; index < model.S * model.U; index ++) {
			if (q < p[index])
				break;
		}
		// index
		vars.s = index / model.U % model.S;
		vars.u = index % model.U;

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
		DOCState docState = model.docStates[docID];
		p = Utils.ensureCapacity(p, docState.numberOfTables);
		fNew = model.gamma * f[model.K];  //  if k = k_new

		// p(x)
		for (k = 0; k < model.K; k++) {
			fNew += model.m_k[k] * f[k];
		}

		// ready to sample table
		for (j = 0; j < docState.numberOfTables; j++) {
			if (docState.wordCountByTable[j] > 0) 
				pSum += docState.wordCountByTable[j] * f[docState.tableToTopic[j]];
			p[j] = pSum;
		}
		pSum += model.alpha * fNew / (model.T + model.gamma); // Probability for t = tNew
		p[docState.numberOfTables] = pSum;
		q = model.random.nextDouble() * pSum;
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
				model.doShuffle();
			nextGibbsSweep();
			
			/*if ((iter > 0) && (iter % 100) == 0) {
				for (int d = 0; d < model.D; d++) {
					DOCState docState = model.docStates[d];
					for (int t = 0; t < docState.wordCountByTable.length; t ++) {
						sampleTopicForTable(d, t);
					}
				}
			}*/
			log.println("iter = " + iter + " #topics = " + model.K + ", #tables = " + model.T );
		}
	}
	
	protected void printF() {
		double sum = 0, ssum = 0;
		for (int k = 0; k < model.K; k ++) {
			sum = 0;
			for (int s = 0; s < model.S; s ++) {
				for (int u = 0; u < model.U; u ++) {
					sum += model.m_k[k]*f_ksu[k][s][u];
					System.out.print(String.format("%1.5f\t", f_ksu[k][s][u]));
				}
			}
			ssum += sum;
			System.out.println(String.format("%1.5f = %1.5f(%2d)", sum, model.m_k[k]*f[k], model.m_k[k]));
		}
		
		sum = 0;
		for (int s = 0; s < model.S; s ++) {
			for (int u = 0; u < model.U; u ++) {
				sum += model.gamma*f_ksu[model.K][s][u];
				System.out.print(String.format("%1.5f\t", f_ksu[model.K][s][u]));
			}
		}
		ssum += sum;
		System.out.println(String.format("%1.5f = %1.5f(%3f)", sum, model.gamma*f[model.K], model.gamma));
		System.out.println(ssum);
	}
	
	public void setConfig(HDPConfig config) {
		this.config = config;
	}
	
	/**
	 * The base of log should be 2 according to the original theory
	 * Now we set the base to e according to the paper of Blei 2003
	 * 
	 * @return
	 */
	public double computePerplexity() {
		double perp = 0.0;

		int N = 0;    // the numbers of all words appeared in the test documents

		model.estimateThetaPhi();
		
		for (int d = 0; d < model.D; d ++) {
			DOCState docState = model.docStates[d];
			for (int n = 0; n < docState.words.length; n ++) {
				int w = docState.words[n].w;

				// k( p(w|z=k) * (p(z=k) )
				double pw = 0.0;
				for (int k = 0; k < model.K; k ++) {
					// log(p(w1)) + log(p(w2)) + ... 
					pw += model.phi[k][w] * model.theta[k][d];
				}
				perp += Math.log(pw);
			}
			N += model.docStates[d].documentLength;
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
		// Configuration
		HDPConfig config = new HDPConfig();
		config.loadConfig(args[0]);
		
		// Dictionary
		Dictionary wDict = new Dictionary();
		Dictionary aDict = new Dictionary();
		Dictionary iDict = new Dictionary();

		// Dataset
		Corpus corpus = new Corpus(wDict, aDict, iDict);
		corpus.readData(new File(config.dir, config.tData), new File(config.dir, config.sData), config.dataCharset);

		// save Dictionaries
		wDict.saveDictionary(config.dir, "dict-v.txt");
		aDict.saveDictionary(config.dir, "dict-a.txt");
		iDict.saveDictionary(config.dir, "dict-i.txt");

		// Model
		HDPModel trainModel = new HDPModel();
		trainModel.setConfig(config);
		trainModel.addInstances(corpus);

		// Training
		HDPRatingTrainer hdp = new HDPRatingTrainer();
		hdp.setConfig(config);
		hdp.setModel(trainModel);

		System.out.println("sizeOfVocabulary = " + trainModel.V);
		System.out.println("totalNumberOfWords = " + trainModel.W);
		System.out.println("NumberOfDocs = " + trainModel.D);

		hdp.run(config.shuffleLag, config.niter, System.out);
		
		System.out.println(hdp.computePerplexity());

		// Save Model
		trainModel.saveModel(config, wDict, aDict);
	}
		
}