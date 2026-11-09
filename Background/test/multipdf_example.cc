#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooChebychev.h"
#include "RooDataHist.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooParametricShapeBinPdf.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooMultiPdf.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "TH1D.h"

using namespace RooFit;
bool shape = true;
void multipdf_example()
{
        // ------------------------------------------------------------
    // Observable
    // ------------------------------------------------------------
    RooRealVar mass("mass","mass",80,180);

    // ------------------------------------------------------------
    // Create binned dataset
    // ------------------------------------------------------------
    TH1D h("h","h",81,80,180);

    // Fill with an exponential distribution: f(x) ~ exp(-0.05 * x)
    for (int i=0; i<10; ++i) {
        std::cout << "\rFilling histogram: " << (i+1) << "/10" << std::flush;
        double x =  (-1./0.05) * log(gRandom->Uniform()); // inverse CDF method
        if (x>=80 && x<=180) h.Fill(x);
    }

    // Convert to RooDataHist
    RooDataHist data("data","data",mass,&h);

    // ------------------------------------------------------------
    // Background model 1: exponential
    // ------------------------------------------------------------
    RooRealVar tau("tau","tau",-0.01,-1.,0.);
    RooExponential bkgExp("bkgExp","bkgExp",mass,tau);

    RooDataHist* rdh1 = dynamic_cast<RooDataHist*>(&data);
    TH1* histo1 = rdh1->createHistogram("massHist1",mass);
    RooArgSet* paramsSet1 = bkgExp.getParameters(data);
    RooArgList paramsList1(*paramsSet1);

    RooParametricShapeBinPdf* shapeExp =
        new RooParametricShapeBinPdf(
            "shapeExp","Parametric exp",
            bkgExp,mass,paramsList1,*histo1
        );

    // ------------------------------------------------------------
    // Background model 2: Chebychev
    // ------------------------------------------------------------
    RooRealVar c0("c0","c0",0.1,-1.,1.);
    RooRealVar c1("c1","c1",-0.1,-1.,1.);
    RooChebychev bkgCheb("bkgCheb","bkgCheb",mass,RooArgList(c0,c1));

    //RooDataHist* rdh2 = dynamic_cast<RooDataHist*>(&data);
    //TH1* histo2 = rdh2->createHistogram("massHist2",mass);
    RooArgSet* paramsSet2 = bkgCheb.getParameters(data);
    RooArgList paramsList2(*paramsSet2);

    RooParametricShapeBinPdf* shapeCheb =
        new RooParametricShapeBinPdf(
            "shapeCheb","Parametric cheby",
            bkgCheb,mass,paramsList2,*histo1
        );

    // ------------------------------------------------------------
    // MultiPdf
    // ------------------------------------------------------------
    RooCategory pdfIndex("pdfIndex","pdfIndex");

    RooArgList pdfList;
    if (shape)    {
        pdfList.add(*shapeExp);
        pdfList.add(*shapeCheb);}
    else    {
        pdfList.add(bkgExp);
        pdfList.add(bkgCheb);
    }
    //pdfList.add(bkgExp);
    //pdfList.add(bkgCheb);

    RooMultiPdf bkgMultiPdf("bkgMultiPdf","bkgMultiPdf",pdfIndex,pdfList);
    RooRealVar bkg_norm("bkgMultiPdf_norm", "bkgMultiPdf_norm", 5000, 0, 5000);
    //RooRealVar shapeExp_norm("shapeExp_norm", "shapeExp_norm", 500, 0, 5000);
    RooRealVar signal_norm("signal_norm", "signal_norm", 10, -5000, 5000);
    signal_norm.setConstant(true); // Fix signal normalization for this example
    RooRealVar signal_shape_norm("signal_shape_norm", "signal_shape_norm", 10, -5000, 5000);
    signal_shape_norm.setConstant(true); // Fix signal normalization for this example
    // ------------------------------------------------------------
    // Signal model (Gaussian)
    // ------------------------------------------------------------
    RooRealVar mean("mean","mean",125,120,130);
    mean.setConstant(true); // Fix mean for this example
    RooRealVar sigma("sigma","sigma",12,0.5,15);
    sigma.setConstant(true); // Fix sigma for this example

    RooGaussian signal("signal","signal",mass,mean,sigma);

    RooArgSet* paramsSet_signal = signal.getParameters(data);
    RooArgList paramsList_signal(*paramsSet_signal);
    RooParametricShapeBinPdf* signal_shape =
        new RooParametricShapeBinPdf(
            "signal_shape","signal shape",
            signal,mass,paramsList_signal,*histo1
        );
    

    // ------------------------------------------------------------
    // Workspace
    // ------------------------------------------------------------
    RooWorkspace w("w","workspace");

    w.import(mass);
    w.import(data);
    w.import(bkg_norm);;
    w.import(bkgMultiPdf);
    w.import(pdfIndex);
    w.import(mean);
    w.import(sigma);

    // IMPORTANT: import the underlying PDFs too
    if (shape)    {
    w.import(*shapeExp);
    //w.import(shapeExp_norm);
    w.import(*shapeCheb);
    w.import(*signal_shape);
    w.import(signal_shape_norm);
    }
    else    {
    w.import(signal_norm);
    w.import(bkgExp);
    w.import(bkgCheb);
    w.import(signal);}

    // ------------------------------------------------------------
    // Write to file
    // ------------------------------------------------------------
    TFile fout("multipdf_workspace_shape.root","RECREATE");
    w.Write();
    fout.Close();
}
