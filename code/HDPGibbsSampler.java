/*
 * Copyright 2011 Arnim Bleier, Andreas Niekler and Patrick Jaehnichen
 * Licensed under the GNU Lesser General Public License.
 * http://www.gnu.org/licenses/lgpl.html
 */

package edu.drexel.sentiment;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
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
 * For more information on the algorithm see:
 * Hierarchical Bayesian Nonparametric Models with Applications. 
 * Y.W. Teh and M.I. Jordan. Bayesian Nonparametrics, 2010. Cambridge University Press.
 * http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/TehJor2010a.pdf
 * 
 * For other known implementations see README.txt
 * 
 * @author <a href="mailto:arnim.bleier+hdp@gmail.com">Arnim Bleier</a>
 */
public class HDPGibbsSampler {
    public final static int INITIAL_TOPIC_SIZE = 20;

	public double beta  = 0.5; // default only
	public double gamma = 1.5;
	public double alpha = 1.0;
	public double lamda = 0.5;
	public double eta = 0.5;
	public double sigma = 0;
	
	public int D;
	public int S = 3;
	public int U = 2;
	public int K = 1;
	public int X = 0;
	public int T;
	public int W;
	public int V;
	
	private Random random = new Random();
	private double[] p;
	private double[] f;
	private double[][][] f_ksu;

	protected DOCState[] docStates;
	//couting variables
	// # of tables by each topic
	protected int[]     m_k;               // m(k.)  m(kd)
	// # of tokens by each topic
	protected int[]     l_k;               // l(k..)
	// # of tokens by each topic and sentiment
	protected int[][]   l_ks;
	// # of tokens by each topic and term
	protected int[][]   l_kw;
	// # of tokens by each topic, sentiment and word
	protected int[][][] l_kws;             // l(ksw)

	// # of tokens by each sentiment for each document
	protected int[][]   l_ds;              // l(kds.)

	//authors
	// # of tokens by each topic and each author with preference u
	protected int[][][]   c_kxu; 

	/**
	 * Initially assign the words to tables and topics
	 * 
	 * @param corpus {@link Corpus} on which to fit the model
	 */
	public void addInstances(Corpus corpus) {
		V = corpus.words2Index.size();
		W = 0;
		D = corpus.docs.size();
		X = corpus.authors2Index.size();

		docStates = new DOCState[D];
		double r_sum = 0, r_mean = 0;
		for (int d = 0; d < D; d ++) {
			docStates[d] = new DOCState(corpus.docs.get(d), d);
			W += corpus.docs.get(d).length;
			r_sum += docStates[d].rating * docStates[d].rating;
			r_mean += docStates[d].rating;
		}

		if (sigma == 0) {
			sigma = r_sum / D - (r_mean / D) * (r_mean / D);
			System.out.println("sigma:" + sigma);
		}

		int k, i, j, s, u;
		DOCState docState;
		p = new double[20]; 
		f = new double[20];
		f_ksu = new double[20][S][U];
		
		m_k = new int[K+1];
		l_k = new int[K+1];
		l_ks = new int[K+1][];
		l_kw = new int[K+1][];
		l_kws = new int[K+1][][];
		l_ds = new int[D][S];

		c_kxu = new int[K+1][][];

		for (k = 0; k <= K; k++) { 	// var initialization done 
			l_ks[k] = new int[S];
			l_kw[k] = new int[V];
			l_kws[k] = new int[V][S];
			c_kxu[k] = new int[X][U];
		}
		for (k = 0; k < K; k++) { 
			docState = docStates[k];
			
			for (i = 0; i < docState.documentLength; i++) {
				s = random.nextInt(S);
				u = random.nextInt(U);
				addWord(docState.docID, i, 0, k, s, u);
			}
		} // all topics have now one document
		
		for (j = K; j < docStates.length; j++) {
			docState = docStates[j]; 
			k = random.nextInt(K);    // all tokens have the same k in each document
			
			for (i = 0; i < docState.documentLength; i++) {
				s = random.nextInt(S); 
				u = random.nextInt(U);
				addWord(docState.docID, i, 0, k, s, u);
			}
		} // the words in the remaining documents are now assigned too
	}

	
	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextGibbsSweep() {
		int t, k, s, u;
		for (int d = 0; d < docStates.length; d++) {
			//System.out.println("------------------------");
			for (int i = 0; i < docStates[d].documentLength; i++) {
				removeWord(d, i); // remove the word i from the state
				computeF(d, i);   // f will be evaluated here
				t = sampleTable(d, i);
				if (t == docStates[d].numberOfTables) { // new Table
					// sampling its Topic and other latent variables
					LatentVariables var = sampleLatentVariables();
					addWord(d, i, t, var.k, var.s, var.u); 
				} else {
					// existing Table
					k = docStates[d].tableToTopic[t];
					LatentVariables var = sampleOtherLatentVariables(k);
					addWord(d, i, t, k, var.s, var.u); 
				}
			}
		}
		defragment();
	}

	protected double[][][] computeF(int d, int i) {
		int k, s, u;
		double vb = V * beta;
		DOCState docState = docStates[d];
		f = Utils.ensureCapacity(f, K);
		p = Utils.ensureCapacity(p, docState.numberOfTables);
		f_ksu = Utils.ensureCapacity(f_ksu, K);

		int x = docState.author;
		int r = docState.rating;
		int w = docState.words[i].w;
		int N = docState.documentLength;

		// r
		double r_sum = 0.0, r_gap = 0.0;
		for (int e = 0; e < N; e++) {
			if (e == i)    // ignore current word
				continue;
			r_sum += docState.words[e].getR();
		}

		// compute f_ksu
		for (k = 0; k < K; k++) {
			f[k] = 0;  //initial 
			for (s = 0; s < S; s++) {
				for (u = 0; u < U; u++) {
					int r_w = WordState.getR(s, u);
					r_gap = r - (r_sum + r_w) / N;
					double r_likelihood = Math.exp((-r_gap * r_gap) / 2 / sigma);

					f_ksu[k][s][u] = (c_kxu[k][x][u] + eta) / (c_kxu[k][x][0] + c_kxu[k][x][1] + 1)
							* (l_kw[k][w] + beta) / (l_k[k] + vb)
							* (l_kws[k][w][s] + lamda) / (l_kw[k][w] + S * lamda)
							* r_likelihood;
					//f_ksu[k][s][u] = (l_ksw[k][s][w] + beta) / (l_ks[k][s] + vb) * eta / S * r_likelihood;

					f[k] += f_ksu[k][s][u];
				}
			}
		}

		// f[K] is f_new
		f[K] = 0;
		for (s = 0; s < S; s++) {
			for (u = 0; u < U; u++) {
				int r_w = WordState.getR(s, u);
				r_gap = r - (r_sum + r_w) / N;
				double r_likelihood = Math.exp((-r_gap * r_gap) / 2 / sigma);

		        f_ksu[K][s][u] = 1.0 / U / V / S * r_likelihood;
		        f[K] += f_ksu[K][s][u];
			}
		}
		//System.out.println(String.format("l_ds[%d](%2d,%2d,%2d)/%d, F[K]:%4.6f", d, l_ds[d][0], l_ds[d][1], l_ds[d][2], N, f[K]));

		return f_ksu;
	}

	private int sampleTopic() {
		double q, pSum = 0.0;
		int k = 0;

		p = Utils.ensureCapacity(p, K);
		for (k = 0; k < K; k++) {
			pSum += m_k[k] * f[k];
			p[k] = pSum;		
		}
		pSum += gamma * f[k];
		p[k] = pSum;
		
		q = random.nextDouble() * pSum;
		for (k = 0; k <= K; k ++) {
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
		LatentVariables vars = new LatentVariables();

		double q, pSum = 0.0;
		int k, s, u;
		p = Utils.ensureCapacity(p, (K + 1) * S * U);

		int index = 0;

		// for existing k
		for (k = 0; k < K; k++) {
			for (s = 0; s < S; s ++) {
				for (u = 0; u < U; u ++) {
					pSum += m_k[k] * f_ksu[k][s][u];

					p[index ++] = pSum;
				}
			}
		}

		// for k_new
		for (s = 0; s < S; s ++) {
			for (u = 0; u < U; u ++) {
				pSum += gamma * f_ksu[k][s][u];

				p[index ++] = pSum;
			}
		}

		q = random.nextDouble() * pSum;
		for (index = 0; index < (K + 1) * S * U; index ++) {
			if (q < p[index])
				break;
		}
		// index
		vars.k = index / U / S;
		vars.s = index / U % S;
		vars.u = index % U;

		return vars;
	}
	
	/**
	 * Decide other latent variables
	 * 
	 * @return the index of the topic
	 */
	private LatentVariables sampleOtherLatentVariables(int k) {
		LatentVariables vars = new LatentVariables();

		double q, pSum = 0.0;
		int s, u;
		p = Utils.ensureCapacity(p, S * U);
		
		int index = 0;

		// for existing k
		for (s = 0; s < S; s++) {
			for (u = 0; u < U; u++) {
				pSum += f_ksu[k][s][u];

				p[index++] = pSum;
			}
		}

		q = random.nextDouble() * pSum;
		for (index = 0; index < S * U; index ++) {
			if (q < p[index])
				break;
		}
		// index
		vars.s = index / U % S;
		vars.u = index % U;

		/*for (int i = 0; i < S * U; i ++) {
			System.out.print(String.format("%6.6f ", p[i]));
		}
		System.out.println(String.format("(%d, %d, %d)", k, vars.s, vars.u));
*/
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
		DOCState docState = docStates[docID];
		p = Utils.ensureCapacity(p, docState.numberOfTables);
		fNew = gamma * f[K];  //  if k = k_new

		// p(x)
		for (k = 0; k < K; k++) {
			fNew += m_k[k] * f[k];
		}

		// ready to sample table
		for (j = 0; j < docState.numberOfTables; j++) {
			if (docState.wordCountByTable[j] > 0) 
				pSum += docState.wordCountByTable[j] * f[docState.tableToTopic[j]];
			p[j] = pSum;
		}
		pSum += alpha * fNew / (T + gamma); // Probability for t = tNew
		p[docState.numberOfTables] = pSum;
		q = random.nextDouble() * pSum;
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
				doShuffle();
			nextGibbsSweep();
			log.println("iter = " + iter + " #topics = " + K + ", #tables = " + T );
		}
	}
		
	
	/**
	 * Removes a word from the bookkeeping
	 * 
	 * @param docID the id of the document the word belongs to 
	 * @param i the index of the word
	 */
	protected void removeWord(int docID, int i){
		DOCState docState = docStates[docID];
		
		int x = docState.author;
		int t = docState.words[i].t;
		int w = docState.words[i].w;
		int k = docState.tableToTopic[t];
		int s = docState.words[i].s;
		int u = docState.words[i].u;

		docState.wordCountByTable[t]--; 
		l_k[k]--; 
		l_ks[k][s]--; 
		l_kw[k][w]--;
		l_kws[k][w][s]--; 
		c_kxu[k][x][u]--;
		l_ds[docID][s]--;

		if (docState.wordCountByTable[t] == 0) { // table is removed
			T--; 
			m_k[k]--; 
			docState.tableToTopic[t] = -1;  // ?
		}
		
		//printCounts(k, s, u, w, x);
	}
	
	/**
	 * Add a word to the bookkeeping
	 * 
	 * @param docID	docID the id of the document the word belongs to 
	 * @param i the index of the word
	 * @param table the table to which the word is assigned to
	 * @param k the topic to which the word is assigned to
	 */
	protected void addWord(int docID, int i, int table, int k, int s, int u) {
		DOCState docState = docStates[docID];
		
		int x = docState.author;
		int w = docState.words[i].w;
		
		docState.words[i].t = table; 
		docState.words[i].s = s;
		docState.words[i].u = u;		

		docState.wordCountByTable[table]++;

		l_k[k] ++; 
		l_ks[k][s] ++;
		l_kw[k][w] ++;
		l_kws[k][w][s] ++;
		c_kxu[k][x][u] ++;
		l_ds[docID][s]++;

		if (docState.wordCountByTable[table] == 1) { // a new table is created
			docState.numberOfTables++;
			docState.tableToTopic[table] = k;
			T ++;
			m_k[k]++; 
			
			docState.tableToTopic = Utils.ensureCapacity(docState.tableToTopic, docState.numberOfTables);
			docState.wordCountByTable = Utils.ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
			if (k == K) { // a new topic is created
				K ++; 
				m_k = Utils.ensureCapacity(m_k, K); 
				l_k = Utils.ensureCapacity(l_k, K);
				l_kw = Utils.add(l_kw, new int[V], K);
				l_ks = Utils.add(l_ks, new int[S], K);
				l_kws = Utils.add(l_kws, new int[V][S], K);
				c_kxu = Utils.add(c_kxu, new int[X][U], K);
			}
		}
	}
	
	protected void printF() {
		double sum = 0, ssum = 0;
		for (int k = 0; k < K; k ++) {
			sum = 0;
			for (int s = 0; s < S; s ++) {
				for (int u = 0; u < U; u ++) {
					sum += m_k[k]*f_ksu[k][s][u];
					System.out.print(String.format("%1.5f\t", f_ksu[k][s][u]));
				}
			}
			ssum += sum;
			System.out.println(String.format("%1.5f = %1.5f(%2d)", sum, m_k[k]*f[k], m_k[k]));
		}
		
		sum = 0;
		for (int s = 0; s < S; s ++) {
			for (int u = 0; u < U; u ++) {
				sum += gamma*f_ksu[K][s][u];
				System.out.print(String.format("%1.5f\t", f_ksu[K][s][u]));
			}
		}
		ssum += sum;
		System.out.println(String.format("%1.5f = %1.5f(%3f)", sum, gamma*f[K], gamma));
		System.out.println(ssum);
	}
	
	/**
	 * For debug
	 * @param k
	 * @param s
	 * @param u
	 * @param w
	 * @param x
	 */
	protected void printCounts(int k, int s, int u, int w, int x) {
		StringBuilder sb = new StringBuilder();
		// Send all output to the Appendable object sb
		Formatter formatter = new Formatter(sb, Locale.US);
		formatter.format("l_k[%1$d]=%6$d\tl_ks[%1$d][%2$d]=%7$d\tl_kw[%1$d][%4$d]=%8$d\tl_ksw[%1$d][%2$d][%4$d]=%9$d\tc_kxu[%1$d][%5$d][%3$d]=%10$d", k, s, u, w, x, l_k[k], l_ks[k][s], l_kw[k][w],l_kws[k][w][s],c_kxu[k][x][u]);
		System.out.println(sb.toString());
		formatter.close();
	}
	
	/**
	 * Removes topics from the bookkeeping that have no words assigned to
	 */
	protected void defragment() {
		int[] kOldToKNew = new int[K];
		int k, newNumberOfTopics = 0;
		for (k = 0; k < K; k++) {
			if (l_k[k] > 0) {
				kOldToKNew[k] = newNumberOfTopics;
				Utils.swap(l_k, newNumberOfTopics, k);
				Utils.swap(m_k, newNumberOfTopics, k);
				Utils.swap(l_kw, newNumberOfTopics, k);
				Utils.swap(l_ks, newNumberOfTopics, k);
				Utils.swap(l_kws, newNumberOfTopics, k);
				Utils.swap(c_kxu, newNumberOfTopics, k);

				newNumberOfTopics++;
			} 
		}
		K = newNumberOfTopics;
		for (int j = 0; j < docStates.length; j++) 
			docStates[j].defragment(kOldToKNew);
	}

	
	/**
	 * Permute the ordering of documents and words in the bookkeeping
	 */
	protected void doShuffle(){
		List<DOCState> h = Arrays.asList(docStates);
		Collections.shuffle(h);
		docStates = h.toArray(new DOCState[h.size()]);
		for (int j = 0; j < docStates.length; j ++){
			List<WordState> h2 = Arrays.asList(docStates[j].words);
			Collections.shuffle(h2);
			docStates[j].words = h2.toArray(new WordState[h2.size()]);
		}
	}
	
	public void setConfig(HDPConfig config) {
		this.gamma = config.gamma;
		this.alpha = config.alpha;
		this.beta = config.beta;
		this.eta = config.eta;
		this.lamda = config.lamda;
		this.sigma = config.sigma;

		this.U = config.U;
		this.S = config.S;
	}

	/**
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		HDPConfig config = new HDPConfig();
		config.loadConfig(args[0]);

		HDPGibbsSampler hdp = new HDPGibbsSampler();
		Corpus corpus = new Corpus();
		corpus.readData(new File(config.dir, config.data), config.dataCharset);
		
		hdp.setConfig(config);

		hdp.addInstances(corpus);

		System.out.println("sizeOfVocabulary = " + hdp.V);
		System.out.println("totalNumberOfWords = " + hdp.W);
		System.out.println("NumberOfDocs = " + hdp.docStates.length);

		hdp.run(config.shuffleLag, config.niter, System.out);

		PrintStream file = new PrintStream(args[1]);
		for (int k = 0; k < hdp.K; k++) {
			for (int w = 0; w < hdp.V; w++)
				file.format("%5d ",hdp.l_kw[k][w]);
			file.println();
		}
		file.close();
		
		int t, x, w, k, s, u, docID;
		double r_sum = 0.0;
		file = new PrintStream("l_ds.txt");
		for (int d = 0; d < hdp.D; d++) {
			DOCState docState = hdp.docStates[d];
			file.format("%5d ",docState.rating);
			r_sum = 0;
			for (int i = 0; i < docState.documentLength; i++) {
				r_sum += docState.words[i].getR();
			}

			// compute f_r
			double r_u = (docState.rating - r_sum / docState.documentLength);
			file.format("%4.3f", r_sum / docState.documentLength);
			file.format("(%4.3f)", r_u);

			for (s = 0; s < hdp.S; s++)
				file.format("%5d ",hdp.l_ds[d][s]);
			file.println();
		}
		file.close();
		
		file = new PrintStream("l_sw.txt");
		file.format("Word    POS    NEG    NEUT"); file.println();
		for (w = 0; w < hdp.V; w++) {
			file.format("%5s ", corpus.index2Words.get(w));
			for (k = 0; k < hdp.K; k++) {
				file.format("%5d:= ", k);
				for(s = 0; s < hdp.S; s ++)
				    file.format("%5.3f  ", (hdp.l_kws[k][w][s] + hdp.beta)/(hdp.l_kw[k][w] + hdp.V*hdp.beta));
			}
			file.println();
		}
		file.close();

		file = new PrintStream(args[2]);
		file.println("d w z t s u x");
		double error = 0.0;
		for (int d = 0; d < hdp.docStates.length; d++) {
			DOCState docState = hdp.docStates[d];
			docID = docState.docID;
			r_sum = 0.0;
			for (int i = 0; i < docState.documentLength; i++) {
				r_sum += docState.words[i].getR();
				
				x = docState.author;
				t = docState.words[i].t;
				w = docState.words[i].w;
				k = docState.tableToTopic[t];
				s = docState.words[i].s;
				u = docState.words[i].u;
				file.println(docID + " " + docState.words[i].w + " " + k + " " + t + " " + s + " " + u + " " + x); 
			}

			// compute f_r
			double r_u = (docState.rating - r_sum / docState.documentLength);
			error += r_u * r_u;
			double r_likelyhood = Math.exp(-(r_u * r_u) / 2 / hdp.sigma);
			System.out.println(String.format("D:%3d  ER:%3.2f  R:%3d  Norm:%3.2f", docID, r_sum / docState.documentLength, docState.rating, r_likelyhood));
		}
		System.out.println("Mean of Square Error:" + error / hdp.D);
		file.close();
		
		file = new PrintStream(args[3]);
		for (x = 0; x < hdp.X; x++) {
			for (k = 0; k < hdp.K; k ++) {
				file.format("%3d  %5d  %5d", x, hdp.c_kxu[k][x][0], hdp.c_kxu[k][x][1]);
				file.println();
			}
		}
		file.close();
	}
		
}