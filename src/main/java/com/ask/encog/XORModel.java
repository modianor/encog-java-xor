package com.ask.encog;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class XORModel {

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
            { 0.0, 1.0 }, { 1.0, 1.0 } };

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };


    public static void main(String[] args) {
        XORModel model = new XORModel();
        BasicNetwork network = model.getNetwork();

        MLDataSet dataSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        final ResilientPropagation train = new ResilientPropagation(network, dataSet);
        int epoch =1;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " error: "+ train.getError());
            epoch++;
        } while(train.getError() > 0.01);

        train.finishTraining();

        System.out.println("Neural Network Results:");
        for(MLDataPair pair: dataSet ) {
            final MLData output = network.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
        }

        Encog.getInstance().shutdown();

    }

    public BasicNetwork getNetwork() {
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false,1));
        network.getStructure().finalizeStructure();
        network.reset();
        return network;
    }
}
