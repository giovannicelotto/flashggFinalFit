// How the number of parameters enters the pdf selection (two separate steps, no double counting):
//
//  1) Within a family (Bernstein n -> n+1, PolExpo, ...): pure NLL, no "+c".
//       chi2_Ftest = 2 (NLL_prev - NLL_this),  prob_Ftest = TMath::Prob(chi2_Ftest, order - prev_order)
//     The difference in the number of parameters is the dof of the chi2 distribution. An order is kept in the
//     envelope only if prob_Ftest < upperEnvThreshold; it then supersedes the previous order of the same family.
//     (chi2_Ftest < 0, i.e. a worse fit at higher order, is set to prob = 0 and passes: check it in the plots.)
//
//  2) Within the envelope (across families, non-nested pdfs): penalised likelihood, "+c".
//       2 NLL + c,  c = number of floating parameters of the pdf (nfloat; in getBestFitFunction 2*bkg->getCorrection())
//     RooMultiPdf::getCorrection() is 0.5 per parameter in NLL units (combine adds it to the NLL) = 1 per parameter in 2NLL units.
//     (getCorrection() counts the observable too, +1 for every pdf: it shifts all pdfs equally, the best pdf does not change.)
//     Used to choose the best-fit index of the multipdf (simplebestFitPdfIndex, getBestFitFunction).
//     The penalty is a flat cost per parameter, instead of a chi2 probability.
//
// The two criteria are not equivalent: for one extra parameter prob < 0.01 requires Delta(2NLL) > 6.6, whereas
// "2 NLL + c" prefers the larger pdf already for Delta(2NLL) > 1.
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"
#include "TFile.h"
#include "TMath.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooWorkspace.h"
#include "TF1.h"
#include "RooDataSet.h"
#include "RooHist.h"
#include "RooAbsData.h"
#include "RooGaussian.h"
#include "RooAbsPdf.h"
#include "RooAddPdf.h"
#include "RooArgSet.h"
#include "RooFitResult.h"
#include "RooMinimizer.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooExtendPdf.h"
#include "RooPolynomial.h"
#include "RooChi2Var.h"
#include "TRandom3.h"
#include "TLatex.h"
#include "TMacro.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TArrow.h"
#include "TKey.h"
#include "RooCategory.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooMultiPdf.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooDoubleCBFast.h"
#include "RooProdPdf.h"
#include "RooGenericPdf.h"
#include "RooNovosibirsk.h"
#include <regex>
#include <algorithm>


#include "HiggsAnalysis/CombinedLimit/interface/RooParametricShapeBinPdf.h"
#include "../interface/PdfModelBuilder.h"
#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFunc.h>
#include <iomanip>
#include "boost/program_options.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/predicate.hpp"

#include "../../tdrStyle/tdrstyle.C"
#include "../../tdrStyle/CMS_lumi.C"
#include "RooChi2Var.h"
#include "RooNumIntConfig.h"
//#include "ROOT/Math/IntegratorOneDimOptions.h"

//ROOT::Math::IntegratorOneDimOptions::SetDefaultIntegrator("Adaptive");
//ROOT::Math::IntegratorOneDimOptions::SetDefaultAbsTolerance(1e-9);
//ROOT::Math::IntegratorOneDimOptions::SetDefaultRelTolerance(1e-9);


using namespace std;
using namespace RooFit;
using namespace boost;




//#include "TROOT.h"
//std::cout << gROOT->GetVersion() << std::endl;

namespace po = program_options;

bool runFtestCheckWithToys=false;
double blind_low = 110;
double blind_high = 140;
double blindFit_low = 110;
double blindFit_high = 140.;
// Mass range configuration
// not configured in main




auto testBias(TH1F htmp, double xlow, double xhigh,
                    double& slope, double& slopeSig, int nPoints, RooHist* hpull) {

  TF1 f0("f0", "pol0", xlow, xhigh);
  TF1 f1("f1", "pol1", xlow, xhigh);

  htmp.Fit(&f0, "RQ0");
  htmp.Fit(&f1, "RQ0");

  double nll0 = 0;
  double nll1 = 0;

  for (int i = 0; i < nPoints; ++i) {
    double x, y;
    hpull->GetPoint(i, x, y);
    if (x < xlow || x > xhigh) continue;

    nll0 += 0.5 * std::pow(y - f0.Eval(x), 2);
    nll1 += 0.5 * std::pow(y - f1.Eval(x), 2);
  }

  slope    = f1.GetParameter(1);
  slopeSig = slope / f1.GetParError(1);

  double prob = TMath::Prob(2 * (nll0 - nll1), 1);
  return prob;
};


RooRealVar *intLumi_ = new RooRealVar("IntLumi","hacked int lumi", 1000.);

TRandom3 *RandomGen = new TRandom3();

RooAbsPdf* getPdf(PdfModelBuilder &pdfsModel, string type, int order, const char* ext=""){
  /* Construct the pdf model using the PdfModelBuilder */
  
  if (type=="Bernstein") return pdfsModel.getBernstein(Form("%s_bern%d",ext,order),order); 
  else if (type=="Exponential") return pdfsModel.getExponentialSingle(Form("%s_exp%d",ext,order),order); 
  else if (type=="PowerLaw") return pdfsModel.getPowerLawSingle(Form("%s_pow%d",ext,order),order); 
  else if ((type=="PolExpo") && (order==1)) return NULL; 
  else if (type=="PolExpo") return pdfsModel.getPolExpo(Form("%s_polexp%d",ext,order),order); 
  else if (type=="PEBern") return pdfsModel.getPEBern(Form("%s_polexpBern%d",ext,order),order); 
  //else if (type=="PowerLawSum") return pdfsModel.getPowerLaw(Form("%s_pow%d",ext,order),order); 
  //else if (type=="ExponentialSum") return pdfsModel.getExponential(Form("%s_expsum%d",ext,order),order); 
  //else if (type=="PowerLawSingle") return pdfsModel.getPowerLawSingle(Form("%s_powsing%d",ext,order),order); 
  //else if (type=="PowerLawGeneric") return pdfsModel.getPowerLawGeneric(Form("%s_powgen%d",ext,order),order); 
  else if (type=="Laurent") return pdfsModel.getLaurentSeries(Form("%s_lau%d",ext,order),order); 
  else if (type=="Chebychev") return pdfsModel.getChebychev(Form("%s_cheb%d",ext,order),order); 
  else {
    cerr << "[ERROR] -- getPdf() -- type " << type << " not recognised." << endl;
    return NULL;
  }
}


struct Chi2Result { double chi2, chi2red; int ndof; double pval; };

  // Restituisce una pdf estesa pronta per il χ² (possibilmente wrappando quella in ingresso)
  // Se trova 'bkg_norm' e/o 'z_norm' li somma e li usa come yield totale; altrimenti scala sui sideband.
  std::unique_ptr<RooAbsPdf> makeExtendedForGOF(RooAbsPdf* pdf, RooRealVar* mass,
                                              RooAbsData* data, const char* rangeList) {
  // Se è già extended (RooAddPdf con yields o RooExtendPdf), usa così com’è
  if (pdf->InheritsFrom("RooExtendPdf")) {
    return std::unique_ptr<RooAbsPdf>((RooAbsPdf*)pdf); // ATTENZIONE: non deletare il pdf originale!
  }
  // Se è RooAddPdf con yields, è comunque extended: non wrappare
  if (pdf->InheritsFrom("RooAddPdf")) {
    // heuristica: se ha una lista di coefficienti che NON somma a 1 è extended; ma non abbiamo accesso semplice qui
    // più semplice: prova a chiamare expectedEvents; se >0, trattala come extended.
    double nexp = pdf->expectedEvents(RooArgSet(*mass));
    if (nexp > 0) return std::unique_ptr<RooAbsPdf>((RooAbsPdf*)pdf);
  }

  // Altrimenti costruiamo una RooExtendPdf con una norma sensata
  // 1) prova a leggere una 'norm' dal modello (es. bkg_norm, z_norm)
  double norm_from_fit = 0.0;
  std::unique_ptr<RooArgSet> pars(pdf->getParameters((const RooArgSet*)0));
  TIterator* it = pars->createIterator();
  while (RooAbsArg* a = (RooAbsArg*)it->Next()) {
    TString n = a->GetName();
    if (n.Contains("bkg_norm") || n.Contains("z_norm") || n.Contains("nbkg")) {
      if (auto* rr = dynamic_cast<RooRealVar*>(a)) norm_from_fit += rr->getVal();
    }
  }
  delete it;

  RooRealVar* normVar = nullptr;
  if (norm_from_fit > 0) {
    normVar = new RooRealVar("__gof_norm","__gof_norm", norm_from_fit, 0., 1e15);
    normVar->setConstant(kTRUE);
  } else {
    // 2) fallback: scala in modo che attesi nei sideband = osservati nei sideband
    std::unique_ptr<RooAbsReal> frac_pdf(
      pdf->createIntegral(RooArgSet(*mass), RooFit::NormSet(RooArgSet(*mass)),
                          RooFit::Range(rangeList))
    );
    RooDataHist* dh = dynamic_cast<RooDataHist*>(data);
    std::unique_ptr<RooDataHist> dh_owner;
    if (!dh) { dh_owner.reset(new RooDataHist("__gof_dh","__gof_dh", RooArgSet(*mass), *data)); dh = dh_owner.get(); }
    double n_sb = dh->sumEntries(nullptr, rangeList);
    double f_sb = frac_pdf ? frac_pdf->getVal() : 0.;
    double norm_val = (f_sb>0) ? (n_sb / f_sb) : n_sb; // robust
    normVar = new RooRealVar("__gof_norm","__gof_norm", norm_val, 0., 1e15);
    normVar->setConstant(kTRUE);
  }
  return std::unique_ptr<RooAbsPdf>(new RooExtendPdf("__gof_ext","__gof_ext", *pdf, *normVar));
}

Chi2Result computeChi2ReducedExtended(RooRealVar* mass, RooAbsPdf* pdf_in, RooAbsData* data,
                                      const char* rangeList) {
    // Step 1: assicurati di avere RooDataHist
    RooDataHist* dh = dynamic_cast<RooDataHist*>(data);
    std::unique_ptr<RooDataHist> dh_owner;
    if (!dh) {
        dh_owner.reset(new RooDataHist("__chi2_dh","__chi2_dh", RooArgSet(*mass), *data));
        dh = dh_owner.get();
    }

    // Step 2: decidi quale pdf usare
    RooAbsPdf* pdf_ext = nullptr;
    std::unique_ptr<RooExtendPdf> owned_ext; // tiene in vita se la creiamo qui

    if (pdf_in->InheritsFrom("RooExtendPdf")) {
        pdf_ext = pdf_in; // già extended → usiamo direttamente
    } else {
        // creo la norm dal fit (se disponibile) o dai sideband
        double norm_val = pdf_in->expectedEvents(RooArgSet(*mass));
        if (norm_val <= 0) {
            // calcola scala dai sideband
            auto frac_pdf = std::unique_ptr<RooAbsReal>(
                pdf_in->createIntegral(RooArgSet(*mass), RooFit::NormSet(RooArgSet(*mass)),
                                       RooFit::Range(rangeList))
            );
            double n_sb = dh->sumEntries(nullptr, rangeList);
            double f_sb = frac_pdf ? frac_pdf->getVal() : 1.0;
            norm_val = (f_sb > 0) ? (n_sb / f_sb) : n_sb;
        }
        auto normVar = new RooRealVar("__gof_norm","__gof_norm", norm_val, 0., 1e15);
        normVar->setConstant(kTRUE);
        owned_ext.reset(new RooExtendPdf("__gof_ext","__gof_ext", *pdf_in, *normVar));
        pdf_ext = owned_ext.get();
    }

    // Step 3: calcola il chi² (Poisson)
    RooChi2Var chi2var("__chi2","__chi2", *pdf_ext, *dh,
                       RooFit::Range(rangeList),
                       RooFit::DataError(RooAbsData::Poisson));
    double chi2 = chi2var.getVal();

    // Step 4: conta bin usati
    auto tmp = std::unique_ptr<RooPlot>(mass->frame());
    dh->plotOn(tmp.get(), RooFit::CutRange(rangeList));
    auto* h = dynamic_cast<RooHist*>(tmp->getObject(int(tmp->numItems()-1)));
    int nbins_used = h ? h->GetN() : 0;

    // Step 5: conta parametri liberi della pdf_ext
    std::unique_ptr<RooArgSet> allPars(pdf_ext->getParameters(*dh));
    RooAbsCollection* floats = allPars->selectByAttrib("Constant", kFALSE);
    int nfloat = floats ? floats->getSize() : 0;
    delete floats;

    int ndof = nbins_used - nfloat;
    if (ndof < 1) ndof = 1;
    double chi2red = chi2 / ndof;
    double pval = TMath::Prob(chi2, ndof);

    return {chi2, chi2red, ndof, pval};
}


void runFit(RooAbsPdf *epdf, RooAbsData *data, double *NLL, int *stat_t, int MaxTries, RooRealVar *mass, RooRealVar *nZ_cat_original, double *chi2_out=nullptr, bool blindSignalRegion=true, int includeTurnOn=-1){
  /* Basic fitting routine, fit is not extended */

  //int nbins = data->numEntries(); 
  //std::cout<< "mass->getMax()"<< mass->getMax() <<std::endl;
  //std::cout<< "mass->getMin()"<< mass->getMin() <<std::endl;
  //std::cout<< "nbins"<< nbins <<std::endl;
  int ntries=0;
  RooArgSet *params_test = epdf->getParameters((const RooArgSet*)(0));
  
  //data->Print("v");
  
  //std::cout<<"\n\n********************************************\n\nInfo on Parameters before fitting\n\n********************************************\n\n";

  //params_test->Print("v");
  int stat=1;
  double minnll=10e8;
  
  while (stat!=0){
    if (ntries>=MaxTries) break;
    RooFitResult *fitTest;
    
    // Choose fit range based on blinding flag
    RooArgSet* params = epdf->getParameters(*data);
    


    TIterator* iter = params->createIterator();
    RooAbsArg* arg;
    
    
    while ((arg = (RooAbsArg*)iter->Next())) {
        if (TString(arg->GetName()).Contains("z_norm")) {
            RooRealVar* z_param = (RooRealVar*)arg;
            z_param->setConstant(true);
            //std::cout << "[INFO] Frozen parameter: " << z_param->GetName() << std::endl;
        }
    }
    if (blindSignalRegion) {
      //std::cout << Form("[INFO] Fitting with signal region  [%d-%d] GeV blinded",(int)blindFit_low,(int)blindFit_high) << std::endl;


      // Save a snapshot of all parameters before doing anything
      delete iter;

      //RooParametric
      //std::cout<<"deleted iter"<<std::endl;
      RooDataHist* rdh = dynamic_cast<RooDataHist*>(data);
      if (!rdh) {
          std::cerr << "Error: data is not a RooDataHist!" << std::endl;
          return;
      }

      TH1* histo = rdh->createHistogram("massHist", *mass);
      RooArgSet* paramsSet = epdf->getParameters(*data);
      RooArgList paramsList(*paramsSet);

      //if (includeTurnOn==1){
        std::cout<<"Turn on isincluded"<<std::endl;
        fitTest =epdf->fitTo(*data,
                            RooFit::Save(1), // Returns a RooFitResult* object
                            RooFit::Minimizer("Minuit2","minimize"),
                            RooFit::Strategy(1),
                            RooFit::PrintLevel(-1),
                            RooFit::Optimize(1),
                            RooFit::SumW2Error(kFALSE),
                            RooFit::Range("blindFit_low,blindFit_high"));
                            //RooFit::Extended(kFALSE),
      //}
      //else{
      //  std::cout<<"Turn on is not included"<<std::endl;
      //  RooParametricShapeBinPdf* shapePdf = new RooParametricShapeBinPdf(
      //    "shapePdf",
      //    "Parametric binned version of epdf",
      //    *epdf,
      //    *mass,
      //    paramsList,
      //    *histo
      //);
      //
      //
      //fitTest = shapePdf->fitTo(*data,
      //                      RooFit::Save(1), // Returns a RooFitResult* object
      //                      RooFit::Minimizer("Minuit2","minimize"),
      //                      RooFit::Strategy(1),
      //                      RooFit::PrintLevel(-1),
      //                      RooFit::Optimize(1),
      //                      RooFit::SumW2Error(kFALSE),
      //                      RooFit::Range("blindFit_low,blindFit_high"));
      //                      //RooFit::Extended(kFALSE),
      //}
      //


      //std::cout<<"\n\n********************************************\n\nInfo on Parameters after fitting\n\n********************************************\n\n";                      
      //fitTest->Print();
      delete histo;



    } else {
        // no blinded regions
        std::cout << "[INFO] Fitting with full mass range" << std::endl;
        fitTest = epdf->fitTo(*data,RooFit::Save(1),RooFit::Minimizer("Minuit2","minimize"),RooFit::Strategy(0),RooFit::PrintLevel(-1),RooFit::Optimize(1),RooFit::SumW2Error(kFALSE));
        }
    
    stat = fitTest->status();
    minnll = fitTest->minNll();
    if (stat!=0) {
      // diagnostics for a failed fit: covariance quality (3 = full accurate, 2 = forced positive definite,
      // 1 = diagonal approximation, 0 = not calculated), edm, parameters at a limit, strong correlations
      std::cout << "[DIAG] " << epdf->GetName() << " try " << ntries << ": status=" << stat
                << " covQual=" << fitTest->covQual() << " edm=" << fitTest->edm()
                << " minNll=" << minnll << std::endl;
      const RooArgList& fl = fitTest->floatParsFinal();
      for (int i = 0; i < fl.getSize(); i++) {
        RooRealVar* v = dynamic_cast<RooRealVar*>(fl.at(i));
        if (!v) continue;
        double range = v->getMax() - v->getMin();
        bool atLimit = (v->getVal() - v->getMin() < 0.01*range) || (v->getMax() - v->getVal() < 0.01*range);
        std::cout << "[DIAG]   " << std::left << std::setw(45) << v->GetName() << std::right
                  << " = " << std::setw(10) << v->getVal() << " +- " << std::setw(9) << v->getError()
                  << "  [" << v->getMin() << ", " << v->getMax() << "]" << (atLimit ? "  <-- AT LIMIT" : "") << std::endl;
      }
      for (int i = 0; i < fl.getSize(); i++) {
        for (int j = i+1; j < fl.getSize(); j++) {
          double rho = fitTest->correlation(fl.at(i)->GetName(), fl.at(j)->GetName());
          if (fabs(rho) > 0.9)
            std::cout << "[DIAG]   correlation " << fl.at(i)->GetName() << " , " << fl.at(j)->GetName() << " = " << rho << std::endl;
        }
      }
    }
    if (stat!=0) params_test->assignValueOnly(fitTest->randomizePars());
    ntries++; 
  }
  *stat_t = stat;
  *NLL = minnll;

  std::cout<<"\n\n********************************************\n\n"<<"NLL No Correction: " <<*NLL<<"\n\n********************************************\n\n";

  // extract chi2
  RooPlot* frame = mass->frame();
  data->plotOn(frame, Name("data"),CutRange("blind_low,blind_high"));
  double Nvis=-1.;
  Nvis = data->sumEntries("1", "blindFit_low,blindFit_high");
  epdf->plotOn(frame, RooFit::Range("blind_low,blind_high"), Name("pdf"), Normalization(Nvis, RooAbsReal::NumEvent), RooFit::NormRange("blindFit_low,blindFit_high"));  // model curve
  RooAbsReal* integral = epdf->createIntegral(
                                          RooArgSet(*mass),  // variables to integrate over
                                          Range("blindFit_low,blindFit_high"),
                                           RooFit::NormSet(RooArgSet(*mass))
                                      );
  std::cout<<"Integral is "<< integral->getVal()<< std::endl;

  double manual_chi2=0;
  RooDataHist* dh = dynamic_cast<RooDataHist*>(data);
  if (!dh) return;

  // create internal histogram
  TH1* h = dh->createHistogram("h_mass", *dynamic_cast<RooRealVar*>(dh->get()->first()));
  int nBinsNotBlinded=0;
  for (int i = 1; i<data->numEntries();i++){

      double low  = h->GetXaxis()->GetBinLowEdge(i);
      double high = h->GetXaxis()->GetBinUpEdge(i);

      double val  = h->GetBinContent(i);
      double err  = h->GetBinError(i);
      double bin_center = (low+high)/2;
      mass->setVal(bin_center);
      double pdfVal = epdf->getVal(RooArgSet(*mass));
      double pdfNormalized = pdfVal*Nvis/integral->getVal()*(h->GetXaxis()->GetBinUpEdge(i)-h->GetXaxis()->GetBinLowEdge(i));// * Nvis / integral->getVal();
      if ((bin_center<blind_low) || (bin_center>blind_high)){
      manual_chi2 += (val - pdfNormalized) * (val - pdfNormalized) /(err*err);
      nBinsNotBlinded++;
  }
  //std::cout<< "Bin center is "<<(low+high)/2<<"  "<<val<<" ± "<<err<<" pdf : "<<pdfNormalized<<std::endl;

}
  
  delete h;




  //TCanvas* c = new TCanvas("c", "Fit result", 800, 600);
  //frame->Draw();

  std::string pdfName = epdf->GetName();
  //if (pdfName.find("Exponential_1") != std::string::npos) {
  //    std::cout << "Name contains Exponential_1\n";
  //    c->SaveAs("/t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/Exp1_cat1.png");
  //} else {
  //    std::cout << "Name does NOT contain Exponential_1\n";
  //}
  
  RooArgSet* params = epdf->getParameters(*data);
  int np = 0;

    for (auto par : *params) {
        RooRealVar* rrvar = dynamic_cast<RooRealVar*>(par);
        if (!rrvar) {
            std::cout << "[DEBUG] Skipping non-RooRealVar parameter" << std::endl;
            continue;
        }

        if (rrvar->isConstant()) {
            //std::cout << "[DEBUG] Skipping constant parameter: "
            //          << rrvar->GetName() 
            //          << " = " << rrvar->getVal()
            //          << std::endl;
            continue;
        }


        np++;
        //std::cout << "[INFO] Floating parameter " << np
        //          << ": " << rrvar->GetName()
        //          << " = " << rrvar->getVal()
        //          << std::endl;
    }
    //std::cout<<"Floating Parameters increased by 1 unit for non-Extended PDF ==> Number of dof is Nbins -  1 since last one is fully determined."<<std::endl;
    np++;
    //std::cout << "[INFO] Total floating parameters: " << np << std::endl;
    //std::cout<<frame->chiSquare("pdf", "data", np)<<std::endl;;
    if (chi2_out){
    *chi2_out = frame->chiSquare("pdf", "data", np);
    cout << "[INFO] Chi2 : nparameters=" << np << ", chi2/ndof=" << *chi2_out << endl;
    std::cout<<"Manual chi2 : "<<manual_chi2<<std::endl;
    std::cout<<"Manual chi2/ndof : "<<manual_chi2/nBinsNotBlinded<<std::endl;
    *chi2_out = manual_chi2/(nBinsNotBlinded-np);

    }
    delete frame;

  }


  
  // DEBUG: Print final parameters after fit
  /*
  if (stat == 0) {
    std::cout << "=== FINAL PARAMETERS AFTER FIT ===" << std::endl;
    RooArgSet *final_params = epdf->getParameters((const RooArgSet*)(0));
    final_params->Print("v");
    
    // Print specific Z-related parameters if they exist
    RooRealVar* z_norm = nullptr;
    RooRealVar* bkg_norm = nullptr;
    
    TIterator* iter = final_params->createIterator();
    RooAbsArg* arg;
    while ((arg = (RooAbsArg*)iter->Next())) {
      if (TString(arg->GetName()).Contains("z_norm")) {
        z_norm = (RooRealVar*)arg;
      }
      if (TString(arg->GetName()).Contains("bkg_norm")) {
        bkg_norm = (RooRealVar*)arg;
      }
    }
    delete iter;
    
    if (z_norm) {
      std::cout << "Z normalization: " << z_norm->getVal() << " +/- " << z_norm->getError() << " events" << std::endl;
      
      if (bkg_norm) {
        std::cout << "Background normalization: " << bkg_norm->getVal() << " +/- " << bkg_norm->getError() << " events" << std::endl;
        double total = z_norm->getVal() + bkg_norm->getVal();
        std::cout << "Z fraction: " << (z_norm->getVal()/total)*100.0 << "%" << std::endl;
      }
    } else {
      // Fallback: look for coefficient approach
      RooRealVar* z_coeff = nullptr;
      TIterator* iter2 = final_params->createIterator();
      RooAbsArg* arg2;
      while ((arg2 = (RooAbsArg*)iter2->Next())) {
        if (TString(arg2->GetName()).Contains("z_coeff")) {
          z_coeff = (RooRealVar*)arg2;
          break;
        }
      }
      delete iter2;
      
      if (z_coeff) {
        std::cout << "Z coefficient: " << z_coeff->getVal() << " +/- " << z_coeff->getError() << std::endl;
      }
    }

    std::cout << "===================================" << std::endl;
  }
  */


double getProbabilityFtest(double chi2, int ndof,RooAbsPdf *pdfNull, RooAbsPdf *pdfTest, RooRealVar *mass, RooAbsData *data, std::string name){
  /* Get probability of the f-test. Currently toys are not used and only simple TMath::Prob is used. */
 
  double prob_asym = TMath::Prob(chi2,ndof);
  if (!runFtestCheckWithToys) return prob_asym;

  int ndata = data->sumEntries();
   
  RooFitResult *fitNullData;// fit the pdfs to the data and keep this fit Result (for randomizing)
  RooFitResult *fitTestData;
  
  fitNullData = pdfNull->fitTo(*data,RooFit::Save(1),RooFit::Strategy(0)
    ,RooFit::Minimizer("Minuit2","minimize"),RooFit::PrintLevel(-1),RooFit::Optimize(1));
  fitTestData = pdfTest->fitTo(*data,RooFit::Save(1),RooFit::Strategy(0)
    ,RooFit::Minimizer("Minuit2","minimize"),RooFit::PrintLevel(-1),RooFit::Optimize(1)); 

  // Ok we want to check the distribution in toys then 
  // Step 1, cache the parameters of each pdf so as not to upset anything 
  RooArgSet *params_null = pdfNull->getParameters((const RooArgSet*)(0));
  RooArgSet preParams_null;
  params_null->snapshot(preParams_null);
  RooArgSet *params_test = pdfTest->getParameters((const RooArgSet*)(0));
  RooArgSet preParams_test;
  params_test->snapshot(preParams_test);
 
  int ntoys = 100;  // Reduced for faster testing
  TCanvas *can = new TCanvas();
  can->SetLogy();
  TH1F toyhist(Form("toys_fTest_%s.pdf",pdfNull->GetName()),";Chi2;",60,-2,10);
  TH1I toyhistStatN(Form("Status_%s.pdf",pdfNull->GetName()),";FitStatus;",8,-4,4);
  TH1I toyhistStatT(Form("Status_%s.pdf",pdfTest->GetName()),";FitStatus;",8,-4,4);

  TGraph *gChi2 = new TGraph();
  gChi2->SetLineColor(kGreen+2);
  double w = toyhist.GetBinWidth(1);

  int ipoint=0;

  for (int b=0;b<0/*toyhist.GetNbinsX()*/;b++){
    double x = toyhist.GetBinCenter(b+1);
    if (x>0){
      gChi2->SetPoint(ipoint,x,(ROOT::Math::chisquared_pdf(x,ndof)));
      ipoint++;
    }
  }
  int npass =0; int nsuccesst =0;
  // mass->setBins(nBinsForFit); // Binning now taken from workspace
  for (int itoy = 0 ; itoy < ntoys ; itoy++){

    params_null->assignValueOnly(preParams_null);
    params_test->assignValueOnly(preParams_test);
    RooDataSet *binnedtoy = pdfNull->generate(RooArgSet(*mass),ndata,0,1);

    int stat_n=1;
    int stat_t=1;
    int ntries = 0;
    double nllNull,nllTest;
    // Iterate on the fit 
    int MaxTries = 2;
    while (stat_n!=0){
      if (ntries>=MaxTries) break;
      RooFitResult *fitNull;
      fitNull = pdfNull->fitTo(*binnedtoy,RooFit::Save(1),RooFit::Strategy(0)
                                              ,RooFit::Minimizer("Minuit2","minimize"),RooFit::Minos(0),RooFit::Hesse(0),RooFit::PrintLevel(-1),RooFit::Optimize(1));
      //,RooFit::Optimize(0));

      nllNull = fitNull->minNll();
      stat_n = fitNull->status();
      if (stat_n!=0) params_null->assignValueOnly(fitNullData->randomizePars());
      ntries++; 
    }
  
    ntries = 0;
    while (stat_t!=0){
      if (ntries>=MaxTries) break;
      RooFitResult *fitTest;
      fitTest = pdfTest->fitTo(*binnedtoy,RooFit::Save(1),RooFit::Strategy(0)
                                              ,RooFit::Minimizer("Minuit2","minimize"),RooFit::Minos(0),RooFit::Hesse(0),RooFit::PrintLevel(-1),RooFit::Optimize(1));
      nllTest = fitTest->minNll();
      stat_t = fitTest->status();
      if (stat_t!=0) params_test->assignValueOnly(fitTestData->randomizePars()); 
      ntries++; 
    }
       
    toyhistStatN.Fill(stat_n);
    toyhistStatT.Fill(stat_t);

    if (stat_t !=0 || stat_n !=0) continue;
    nsuccesst++;
    double chi2_t = 2*(nllNull-nllTest);
    if (chi2_t >= chi2) npass++;
        toyhist.Fill(chi2_t);

  } // end loop over toys

  double prob=0;
  if (nsuccesst!=0)  prob = (double)npass / nsuccesst;
  toyhist.Scale(1./(w*toyhist.Integral()));
  toyhist.Draw();
  TArrow lData(chi2,toyhist.GetMaximum(),chi2,0);
  lData.SetLineWidth(2);
  lData.Draw();
  gChi2->Draw("L");
  TLatex *lat = new TLatex();
  lat->SetNDC();
  lat->SetTextFont(42);
  lat->DrawLatex(0.1,0.91,Form("Prob (asymptotic) = %.4f (%.4f)",prob,prob_asym));
  std::cout << "debug get probability f test: " << name.c_str() << std::endl; 
  can->SaveAs(name.c_str());

  TCanvas *stas =new TCanvas();
  toyhistStatN.SetLineColor(2);
  toyhistStatT.SetLineColor(1); 
  TLegend *leg = new TLegend(0.2,0.6,0.4,0.87); leg->SetFillColor(0);
  leg->SetTextFont(42);
  leg->AddEntry(&toyhistStatN,"Null Hyp","L");
  leg->AddEntry(&toyhistStatT,"Test Hyp","L");
  toyhistStatN.Draw();
  toyhistStatT.Draw("same");
  leg->Draw();
  stas->SaveAs(Form("%s_fitstatus.pdf",name.c_str()));
  //reassign params
  params_null->assignValueOnly(preParams_null);
  params_test->assignValueOnly(preParams_test);

  delete can; delete stas;
  delete gChi2;
  delete leg;
  delete lat;

  // Still return the asymptotic prob (usually its close to the toys one)
  return prob_asym;

}

double getGoodnessOfFit(RooAbsPdf* pdf,RooAbsData *data,  int nBinsNotBlinded, double chi2FromFit=-1. ){
  RooRealVar nTot("nTot","total expected yield", data->sumEntries(), 0, 1e6);
  RooExtendPdf* epdf = new RooExtendPdf("epdf","extended version of pdf", *pdf, nTot);
  RooArgSet* params = epdf->getParameters(*data);
  int ndof = 0;
  int np = 0;

  for (auto par : *params) {
      RooRealVar* rrvar = dynamic_cast<RooRealVar*>(par);
      if (!rrvar) continue;
      if (rrvar->isConstant()) continue;                  // skip fixed
      //if (std::string(rrvar->GetName()) == "norm") continue;  // skip bookkeeping norm
  
      np++;
      //std::cout << "[INFO] Floating parameter " << np
      //          << ": " << rrvar->GetName()
      //          << " = " << rrvar->getVal()
      //          << std::endl;
  }
  ndof = nBinsNotBlinded - np;
  double prob = TMath::Prob(chi2FromFit*ndof, ndof);
  std::cout<<"--Chi2FromFit : "<< chi2FromFit << " ndof : "<< ndof << "prob : "<< prob;
  
  return prob;

}

void plotComponents(RooRealVar *mass, RooAbsPdf *pdf,RooAbsPdf *zModel, RooAbsData *data, int nBinsNotBlinded, RooRealVar *nZ_original, string name,vector<string> flashggCats_, int status, double *prob, double NLL, int *pullsStatus, double chi2FromFit=-1, bool fitWasBlinded=false, int nBins=-1 ){
  std::cout << "Plot Components : " << nBinsNotBlinded << std::endl;
  /* Plot pdf vs data with separate Z component and ratio plot */
    
  // Calculate goodness of fit
  
  *prob = getGoodnessOfFit(pdf,data,nBinsNotBlinded,chi2FromFit);

  
  RooPlot *plot = mass->frame();
  
  // Blinding: escludi solo la SR [100,135] GeV
  if (fitWasBlinded) {
    std::cout << "[INFO] Plotting with blinded signal region [" << blind_low << ", " << blind_high << "] GeV" << std::endl;
    mass->setRange("blind_low", mass->getMin(), blind_low);   // Sideband basso
    mass->setRange("blind_high", blind_high, mass->getMax()); // Sideband alto
    mass->setRange("blindFit_low", mass->getMin(), blindFit_low);     // Low sideband: min-100 GeV
    mass->setRange("blindFit_high", blindFit_high, mass->getMax());   // High sideband: 135-max GeV
  } else {
    std::cout << "[INFO] Plotting with full mass range (no blinding)" << std::endl;
    mass->setRange("blind_low", mass->getMin(), blind_high);   // Sideband basso
    mass->setRange("blind_high", blind_high, mass->getMax()); // Sideband alto
    mass->setRange("blindFit_low", mass->getMin(), blindFit_high);     // Low sideband: min-100 GeV
    mass->setRange("blindFit_high", blindFit_high, mass->getMax());   // High sideband: 135-max GeV
  }

  // Plot data con blinding SR se richiesto
  double Nvis=-1;

  if (fitWasBlinded) {
    data->plotOn(plot, MarkerStyle(20), MarkerSize(0.8), CutRange("blind_low,blind_high"));
    Nvis = data->sumEntries("1", "blindFit_low,blindFit_high");
  } else {
    data->plotOn(plot, MarkerStyle(20), MarkerSize(0.8));
    Nvis = data->sumEntries();  // full range
    pdf->plotOn(plot, LineColor(kRed), LineWidth(2),
        Normalization(Nvis, RooAbsReal::NumEvent));
    // No NormRange needed — full range normalization is the default
}

  // Normalizzazione: conta solo eventi nei sidebands
  

  
  pdf->plotOn(plot,
              LineColor(kRed),LineWidth(2), 
              Normalization(Nvis, RooAbsReal::NumEvent),
              RooFit::NormRange("blindFit_low,blindFit_high"));
  // Plot PDF su tutto il range con normalizzazione naturale
  std::cout << "[INFO] Plotting function over full mass range with natural normalization" << std::endl;
  //epdf->plotOn(plot,LineColor(kRed),LineWidth(2),Name("total"),
  //                  NormRange("blind_low,blind_high"),
  //                  Range(mass->getMin(),mass->getMax()));
  //RooRealVar* nTot = (RooRealVar*) pdf->getParameters(*data)->find("nTot");


  // Componenti: stessa cosa
  if (pdf->InheritsFrom("RooAddPdf")) {
    RooAddPdf* addPdf = (RooAddPdf*)pdf;
    RooArgList comps = addPdf->pdfList();
    if (comps.getSize() >= 2) {
      RooAbsPdf* bkgPdf = (RooAbsPdf*)comps.at(1);    // components 0 is Z to bb, component 1 is bkg
      if (bkgPdf) {
        // Usa sempre la normalizzazione del fit (bkg_norm->getVal()) e NormRange coerente
        pdf->plotOn(plot,Components(*bkgPdf),LineColor(kBlue),LineStyle(kDashed),LineWidth(2),Name("background"),
                                 Normalization(Nvis, RooAbsReal::NumEvent),
                                 RooFit::NormRange("blindFit_low,blindFit_high"));

      //                   Range(mass->getMin(),mass->getMax()));

      }
      
      // Plot Z component with proper normalization
      if (zModel) {
        // Anche la Z viene normalizzata con Nvis e NormRange coerente
        pdf->plotOn(plot,Components(*zModel),LineColor(kGreen+2),LineWidth(2),Name("Z"),
                                 Normalization(Nvis, RooAbsReal::NumEvent),
                                 RooFit::NormRange("blindFit_low,blindFit_high"));
      }
    }
  }

  // Use chi2 from fit if provided
  double chi2;
  if (chi2FromFit > 0) {
    chi2 = chi2FromFit;
    cout << "[INFO] Using chi2 from fit: " << chi2 << endl;
  } else {
    int np = pdf->getParameters(*data)->getSize();
    chi2 = plot->chiSquare(np);
    cout << "[WARNING] Using chi2 from plot (may be unreliable): " << chi2 << endl;
  }

  TCanvas *canv = new TCanvas("canv","canv",800,800);
  canv->Divide(1,2);
  
  // Upper plot: data and fits
  canv->cd(1);
  gPad->SetLeftMargin(0.15);
  gPad->SetBottomMargin(0.02);
  gPad->SetPad(0.01,0.3,0.99,0.99);
  plot->GetXaxis()->SetTitleSize(0.04);
  plot->GetXaxis()->SetTitle("m_{jj} [GeV]");
  plot->GetXaxis()->SetLabelSize(0);
  plot->GetYaxis()->SetTitleSize(0.05);
  plot->GetYaxis()->SetLabelSize(0.04);
  plot->GetYaxis()->SetTitleOffset(1.2);
  plot->GetYaxis()->SetTitle("Events");
  
  plot->SetTitle("");
  plot->SetMaximum(plot->GetMaximum()*1.3);
  plot->Draw();
  
  // Add legend
  TLegend *leg = new TLegend(0.65,0.65,0.89,0.89);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);
  leg->AddEntry("Graph","Data","pe");
  leg->AddEntry("total","Total fit","l");

  // Extract clean pdf name (same logic as plotMultiPdf)
  std::string pdfFullName = pdf->GetName();
  std::string cleanLabel = pdfFullName;
  std::string prefix = "env_pdf_";
  size_t start = pdfFullName.find(prefix);
  if (start != std::string::npos) {
      start += prefix.length();
      size_t end = pdfFullName.find("_with", start);
      if (end == std::string::npos) end = pdfFullName.length();
      cleanLabel = pdfFullName.substr(start, end - start);
      std::replace(cleanLabel.begin(), cleanLabel.end(), '_', ' ');
      std::regex catRegex("cat[0-9]+");
      cleanLabel = std::regex_replace(cleanLabel, catRegex, "");
      cleanLabel = std::regex_replace(cleanLabel, std::regex("\\s+"), " ");
      // trim trailing space
      if (!cleanLabel.empty() && cleanLabel.back() == ' ')
          cleanLabel.pop_back();
  }


  leg->AddEntry("background", cleanLabel.c_str(), "l");
  leg->AddEntry("Z", "Z (r_{Z}=1)", "l");
  //leg->AddEntry("Z MC","Z MC","l");
  leg->Draw();
  
  TLatex *lat = new TLatex();
  lat->SetNDC();
  lat->SetTextFont(42);
  lat->SetTextSize(0.034);
  lat->DrawLatex(0.15,0.94,Form("#chi^{2}/ndof = %.3f, Prob = %.2f, #Lambda_{corr}= %.1f",chi2,*prob,2*NLL));

  // Lower plot: ratio (Data-Background)/Background to highlight Z contribution
  canv->cd(2);
  gPad->SetLeftMargin(0.15);
  gPad->SetBottomMargin(0.35);
  gPad->SetPad(0.01,0.01,0.99,0.3);
  gPad->SetGridy();
  
  RooPlot *ratioPlot = mass->frame();
  
  // Create pull plot with conditional blinding
  if (fitWasBlinded) {
    data->plotOn(ratioPlot,MarkerStyle(20),MarkerSize(0.8),CutRange("blind_low,blind_high"));
    pdf->plotOn(ratioPlot,LineColor(kRed),LineWidth(2),Normalization(Nvis, RooAbsReal::NumEvent),RooFit::NormRange("blindFit_low,blindFit_high")); // Let RooFit handle natural normalization
  } else {
    data->plotOn(ratioPlot,MarkerStyle(20),MarkerSize(0.8));
    pdf->plotOn(ratioPlot,LineColor(kRed),LineWidth(2), Normalization(Nvis, RooAbsReal::NumEvent), RooFit::NormRange("blindFit_low,blindFit_high")); // Let RooFit handle natural normalization
  }
  
  // Calculate pulls excluding signal region
  RooHist* hpull = ratioPlot->pullHist("", "", true); // Calculate pulls
  hpull->SetMarkerSize(0.6);
  
  
  

  //std::cout<<"LOOK HERE " << std::endl;
  // 1) Create a TH1 with same binning as mass
  int nPoints = hpull->GetN();
  double xmin = mass->getMin();
  double xmax = mass->getMax();

  TH1F htmp("htmp", "rebinned from RooHist", nPoints, xmin, xmax);


  for (int i = 0; i < nPoints; i++) {
      double x, y;
      hpull->GetPoint(i, x, y);
      htmp.SetBinContent(htmp.FindBin(x), y);
      }
  htmp.Sumw2(false);
  for (int b = 1; b <= htmp.GetNbinsX(); ++b){
    htmp.SetBinError(b, 1.0);
    }
  double slope1, sig1;
  double pWide = testBias(htmp, xmin, 110, slope1, sig1, nPoints, hpull);

  //double slope2, sig2;
  //double pEdge = testBias(htmp, blind_low - 20., blind_low, slope2, sig2, nPoints, hpull);
  //std::cout << "Bias Test Results:" << std::endl;
  //std::cout << " Wide Region (" << xmin << " to " << blind_low << "): p-value = " << pWide << ", slope = " << slope1 << ", significance = " << sig1 << std::endl;
  //std::cout << " Edge Region (" << blind_low - 20. << " to " << blind_low << "): p-value = " << pEdge << ", slope = " << slope2 << ", significance = " << sig2 << std::endl;
  bool biasWide = false;//pWide < 0.01;
  bool biasWide_slope =  sig1 > 2.0;
  bool biasEdge = false;//(pEdge < 0.05 && sig2 > 1.645);

  //*pullsStatus = (biasWide || biasEdge);

    
  if (biasWide) {
    //std::cout << "[BIAS DETECTED] Wide Region: p-value = " << pWide << ", slope = " << slope1 << ", significance = " << sig1 << std::endl;
    *pullsStatus = (int) pow(2,0); // bias detected
  }
  if (biasWide_slope) {
    //std::cout << "[BIAS DETECTED] Wide Region: p-value = " << pWide << ", slope = " << slope1 << ", significance = " << sig1 << std::endl;
    *pullsStatus = *pullsStatus+(int)pow(2,1); // bias detected
  }
  if (biasEdge) {
    //std::cout << "[BIAS DETECTED] Edge Region: p-value = " << pEdge << ", slope = " << slope2 << ", significance = " << sig2 << std::endl;
    *pullsStatus = 100; // bias detected
  }
  if ((biasWide) || (biasEdge)){
    //*pullsStatus = 1; // bias detected
    canv->cd(1);
    TLatex *lat_Bias = new TLatex();
    lat_Bias->SetNDC();
    lat_Bias->SetTextFont(42);
    lat_Bias->SetTextSize(0.034);
    lat_Bias->DrawLatex(0.8,0.94,Form("LowMass Bias= %d",*pullsStatus));
  }
  else{
    *pullsStatus = 0; // no bias
  }


  canv->cd(2);
  
  RooPlot *pullPlotFinal = mass->frame();
  pullPlotFinal->addPlotable(hpull,"P");
  
  pullPlotFinal->GetYaxis()->SetNdivisions(504);
  pullPlotFinal->GetYaxis()->SetLabelSize(0.12);
  pullPlotFinal->GetYaxis()->SetTitleSize(0.12);
  pullPlotFinal->GetYaxis()->SetTitleOffset(0.5);
  pullPlotFinal->GetYaxis()->SetTitle("Pull");
  pullPlotFinal->GetXaxis()->SetTitleSize(0.12);
  pullPlotFinal->GetXaxis()->SetLabelSize(0.12);
  pullPlotFinal->GetXaxis()->SetTitle("m_{jj} (GeV)");
  pullPlotFinal->SetTitle("");
  pullPlotFinal->SetMaximum(3);
  pullPlotFinal->SetMinimum(-3);
  pullPlotFinal->Draw();

  //canv->SaveAs(Form("%s_components.pdf",name.c_str()));
  canv->SaveAs(Form("%s_components.png",name.c_str()));

  delete canv;
  delete lat;
  delete leg;
}

void plot1(RooRealVar *mass, RooAbsPdf *pdf, RooAbsData *data, int nBinsNotBlinded, string name,vector<string> flashggCats_, int status, double *prob, double chi2FromFit=-1, bool fitWasBlinded=false, int nBins=-1 ){
  /* Plot single pdf vs data, with chi2 from fit */
    
  // Calculate goodness of fit
  // here is the computation of prob
  // redoit
  *prob = getGoodnessOfFit(pdf,data,nBinsNotBlinded,chi2FromFit);
  // Use the data directly without conversion to preserve workspace binning
  RooPlot *plot = mass->frame();
  
  // Blinding: escludi solo la SR [100,135] GeV
  mass->setRange("blind_low", mass->getMin(), blind_low);   // Sideband basso
  mass->setRange("blind_high", blind_high, mass->getMax()); // Sideband alto
  
  // Fit: usa solo sidebands
  mass->setRange("blindFit_low", mass->getMin(), blindFit_low);     // Low sideband: mass->getMin()-100 GeV
  mass->setRange("blindFit_high", blindFit_high, mass->getMax());   // High sideband: 135-mass->getMax() GeV
  
  // Plot data con blinding SR se richiesto
  double Nvis = -1.;
  if (fitWasBlinded) {
    data->plotOn(plot, MarkerStyle(20), MarkerSize(0.8),
                     CutRange("blind_low,blind_high"));
    Nvis = data->sumEntries("1", "blindFit_low,blindFit_high");
  } else {
    data->plotOn(plot, MarkerStyle(20), MarkerSize(0.8));
  }


  // Plot PDF su tutto il range con normalizzazione naturale
  //std::cout << "[INFO] Plotting function over full mass range with natural normalization" << std::endl;
  pdf->plotOn( plot,
                LineColor(kCyan),
                LineWidth(2),
                Normalization(Nvis, RooAbsReal::NumEvent),
                RooFit::NormRange("blindFit_low,blindFit_high"));

 
  
  // Use chi2 from fit if provided, otherwise calculate from plot
  double chi2;
  if (chi2FromFit > 0) {
    chi2 = chi2FromFit;
    cout << "[INFO] Using chi2 from fit: " << chi2 << endl;
  } else {
    // Fallback: calcolare dal grafico (può essere inaffidabile per RooDataHist)
    int np = pdf->getParameters(*data)->getSize();
    chi2 = plot->chiSquare(np);
    cout << "[WARNING] Using chi2 from plot (may be unreliable): " << chi2 << endl;
  }

  TCanvas *canv = new TCanvas();
  // canv->Divide(1,2); // Disabilita il grafico di pull per ora
  // canv->cd(1);
  gPad->SetLeftMargin(0.15);
  gPad->SetBottomMargin(0.10);
  // gPad->SetPad(0.01,0.2,0.99,0.99);
  plot->GetXaxis()->SetTitleSize(0.04);
  plot->GetXaxis()->SetTitle("m_{jj} (GeV)");
  plot->GetYaxis()->SetTitleSize(0.04);;
  //plot->GetYaxis()->SetTitle("Entries");
  plot->GetYaxis()->SetTitleOffset(1.35);

  // PDF già aggiunta sopra, aggiungi solo i parametri
  pdf->paramOn(plot,RooFit::Layout(0.34,0.85,0.89),RooFit::Format("NEA",AutoPrecision(2)));
  plot->getAttText()->SetTextSize(0.025);
  plot->SetMaximum(plot->GetMaximum()*1.4);
  plot->SetTitle("");
  plot->Draw();
  TLatex *lat = new TLatex();
  lat->SetNDC();
  lat->SetTextFont(42);
  lat->SetTextSize(0.034);
  lat->DrawLatex(0.15,0.94,Form("#chi^{2}/ndof = %.3f, Prob = %.2f, Fit Status = %d ",chi2,*prob,status));

  TLatex *lab = new TLatex();
  lab->SetNDC();
  lab->SetTextFont(42);
  lab->SetTextSize(0.034);
  std::string delimiter = "/";
  size_t pos = 0;
  std::string s = name;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    //std::cout << token << std::endl;
    s.erase(0, pos + delimiter.length());
  }
  //delimiter = ".";
  //token = s.substr(0, s.find(delimiter));
  //std::cout << s << std::endl;
  lab->DrawLatex(0.55,0.25,s.c_str());

  /* TODO: Fix pull calculation for RooDataHist
  // pulls
  canv->cd(2);
  gPad->SetLeftMargin(0.15);
  gPad->SetPad(0.01,0.01,0.99,0.2);
  gPad->SetGridy();
  
  // Calculate pulls explicitly using named objects
  RooHist* hpull = plot->pullHist("data", "pdf");
  
  RooPlot *plot2 = mass->frame();
  plot2->GetYaxis()->SetNdivisions(504);
  plot2->GetYaxis()->SetLabelSize(0.17);
  plot2->GetYaxis()->SetTitleSize(0.17);
  plot2->GetYaxis()->SetTitleOffset(0.24);
  //plot2->GetYaxis()->SetRangeUser(-3.0,3.0);
  plot2->SetMinimum(-3.);
  plot2->SetMaximum(3.);
  plot2->GetYaxis()->SetTitle("Pulls") ;
  plot2->GetXaxis()->SetTitle("");
  plot2->GetXaxis()->SetLabelOffset(5);
  plot2->addPlotable(hpull,"P"); 
  plot2->Draw();
  */

  //canv->SaveAs(Form("%s.pdf",name.c_str()));
  canv->SaveAs(Form("%s.png",name.c_str()));

  delete canv;
  delete lat;
}
void plotMultiPdf(RooRealVar *mass, RooMultiPdf *pdfs, RooCategory *catIndex, RooAbsData *data, string name, vector<string> flashggCats_, int cat, int bestFitPdf=-1, bool fitWasBlinded=true){
  /* Plot MultiPdf vs data */
  
  int color[7] = {kCyan+1,kGreen+2,kAzure+2,kTeal+1,kSpring+2,kCyan+3,kGreen+1};
  TLegend *leg = new TLegend(0.35,0.65,0.95,0.90);
  leg->SetFillColor(0);
  leg->SetLineColor(1);
  RooPlot *plot = mass->frame();

  // Blinding: escludi solo la SR [100,135] GeV
  mass->setRange("blind_low", mass->getMin(), blind_low);   // Sideband basso
  mass->setRange("blind_high", blind_high, mass->getMax()); // Sideband alto
  
  // Fit: usa solo sidebands
  mass->setRange("blindFit_low", mass->getMin(), blindFit_low);     // Low sideband: mass->getMin()-100 GeV
  mass->setRange("blindFit_high", blindFit_high, mass->getMax());   // High sideband: 135-mass->getMax() GeV
  
  data->plotOn(plot,MarkerStyle(20),MarkerSize(0.8),CutRange("blind_low,blind_high")); // Plot escludendo la regione di segnale
  
  TCanvas *canv = new TCanvas();
  //TPad *pad1 = new TPad("pad1","pad1",0,0,1,1);
  //pad1->SetBottomMargin(0.18);
  //pad1->Draw();
  //pad1->cd();

  int currentIndex = catIndex->getIndex();
  TObject *datLeg = plot->getObject(int(plot->numItems()-1));
  leg->AddEntry(datLeg,"Data");
  int style=1;
  //RooAbsPdf *pdf;
  //RooCurve *nomBkgCurve;
  double Nvis;
  if (fitWasBlinded) {
    Nvis = data->sumEntries("1", "blindFit_low,blindFit_high");
    std::cout << "[INFO] Blinded fit: Nvis = " << Nvis << " events in sidebands" << std::endl;
  } else {
    Nvis = data->sumEntries();
    std::cout << "[INFO] Unblinded fit: Nvis = " << Nvis << " events total" << std::endl;
  }
  //int bestcol= -1;
  for (int icat=0;icat<catIndex->numTypes();icat++){
    int col;
    if (icat<=6) col=color[icat];
    else {col=kBlack; style++;}
    catIndex->setIndex(icat);
    //pdfs->getCurrentPdf()->fitTo(*data,RooFit::Minos(0),RooFit::Minimizer("Minuit2","minimize"),RooFit::Strategy(0),RooFit::PrintLevel(-1),RooFit::Optimize(1));  
    if (fitWasBlinded) pdfs->getCurrentPdf()->plotOn(plot,LineColor(col),
                                                          Normalization(Nvis, RooAbsReal::NumEvent), // fix blinded to full range norm
                                                          RooFit::NormRange("blindFit_low,blindFit_high"));
                                                          //LineWidth(2),
                                                          //Range(mass->getMin(),mass->getMax()),
    else pdfs->getCurrentPdf()->plotOn(plot,LineColor(col),LineStyle(style),Range(mass->getMin(),mass->getMax()),
                     Normalization(Nvis, RooAbsReal::NumEvent)); // Let RooFit handle natural normalization
  //  pdfs->getCurrentPdf()->plotOn(plot,LineColor(col),LineStyle(style),RooFit::Range("full"),RooFit::NormRange("full"));
    TObject *pdfLeg = plot->getObject(int(plot->numItems()-1));
    std::string ext = "";
    if (bestFitPdf==icat) {
      ext=" (Best Fit Pdf) ";
      //pdf= pdfs->getCurrentPdf();
      //nomBkgCurve = (RooCurve*)plot->getObject(plot->numItems()-1);
      //bestcol = col;
    }
    string pdfName = pdfs->getCurrentPdf()->GetName();
    std:: cout << "PDF Name: " << pdfName << std::endl; 
    std::string prefix = "env_pdf_";
    size_t start = pdfName.find(prefix) + prefix.length();
    size_t end = pdfName.find("_with", start);
    std::string family = pdfName.substr(start, end - start);
    std::string cleanLabel = family;

  // sostituisce '_' con spazio
  std::replace(cleanLabel.begin(), cleanLabel.end(), '_', ' ');

  // rimuove "cat%d" se presente
  std::regex catRegex("cat[0-9]+");
  cleanLabel = std::regex_replace(cleanLabel, catRegex, "");

  // rimuove eventuali doppi spazi
  cleanLabel = std::regex_replace(cleanLabel, std::regex("\\s+"), " ");
  leg->AddEntry(pdfLeg, Form("%s%s", cleanLabel.c_str(), ext.c_str()), "L");
  }
  leg->SetTextSize(0.040);      // testo più grande
  leg->SetEntrySeparation(0.15); // più spazio tra le righe
  plot->SetTitle(Form("Category 2"));
  plot->SetMaximum(plot->GetMaximum()*1.4);
  plot->GetXaxis()->SetTitle("m_{jj} (GeV)");
  plot->Draw();
  //plot->GetYaxis()->SetTitle("Events / nGeV");
  plot->GetYaxis()->SetTitleOffset(1.6); // default ~1.2

  leg->Draw("same");
  CMS_lumi( canv, 0, 0);
  //canv->SaveAs(Form("%s.pdf",name.c_str()));
  canv->SaveAs(Form("%s.png",name.c_str()));
  catIndex->setIndex(currentIndex);
  delete canv;
}


void transferMacros(TFile *inFile, TFile *outFile){
  
  TIter next(inFile->GetListOfKeys());
  TKey *key;
  while ((key = (TKey*)next())){
    if (string(key->ReadObj()->ClassName())=="TMacro") {
      //cout << key->ReadObj()->ClassName() << " : " << key->GetName() << endl;
      TMacro *macro = (TMacro*)inFile->Get(key->GetName());
      outFile->cd();
      macro->Write();
    }
  }
}

int getBestFitFunction(RooMultiPdf *bkg, RooAbsData *data, RooCategory *cat, RooRealVar *mass, RooRealVar *nZ_cat_original, bool silent=false, int includeTurnOn=-1, bool blindSignalRegion=true){
  /* Get index of the best fit pdf (minimum NLL, including correction) among functions in the multipdf.
   All fits are performed again. */

  double global_minNll = 1E10;
  int best_index = 0;
  int number_of_indeces = cat->numTypes();
    
  RooArgSet snap,clean;
  RooArgSet *params = bkg->getParameters((const RooArgSet*)0);
  params->remove(*cat);  // pdf_index is RooCategory, removed from parameters
  params->snapshot(snap);
  params->snapshot(clean);
  if (!silent) {
    //params->Print("V");
  }
 
  // Uncomment to try to make converge a failed _index=pdfindex_0_2016_13TeVfit
  //bkg->setDirtyInhibit(1);
  //RooAbsReal *nllm = bkg->createNLL(*data);
  //RooMinimizer minim(*nllm);
  //minim.setStrategy(1);
  
  for (int id=0;id<number_of_indeces;id++){    
    params->assignValueOnly(clean);
    cat->setIndex(id);

    //RooAbsReal *nllm = bkg->getCurrentPdf()->createNLL(*data);

    if (!silent) {
      /*
      std::cout << "BEFORE  MAKING FIT" << std::endl;
      params->Print("V");
      std::cout << "-----------------------" << std::endl;    
      */
    }
    
    //minim.minimize("Minuit2","minimize");
    double minNll=0; //(nllm->getVal())+bkg->getCorrection();
    int fitStatus=1;    
    //RooRealVar nTot("nTot","total expected yield", data->sumEntries(), 0, 1e6);
    //RooExtendPdf* ebkgpdf = new RooExtendPdf("epdf","extended version of pdf", *(bkg->getCurrentPdf()), nTot);
    std::cout<<"v1"<<std::endl;
    runFit(bkg->getCurrentPdf(),data,&minNll,&fitStatus,3,mass,  nZ_cat_original,  nullptr, blindSignalRegion, includeTurnOn);
    // Add the penalty. RooMultiPdf::getCorrection() is 0.5 per parameter in NLL units (cFactor = 0.5), i.e. combine adds it to the NLL.
    // Here we compare 2*NLL, so the penalty is c = 2*getCorrection() = 1 per parameter (same as 2NLL + nfloat in the family loop).
    minNll = 2.*(minNll + bkg->getCorrection());
    std::cout << "[INFO] AFTER FITTING" << std::endl;
    std::cout << "[INFO] Function was " << bkg->getCurrentPdf()->GetName() <<std::endl;
    std::cout << "[INFO] Correction Applied (NLL units, 0.5 per parameter) is " << bkg->getCorrection() <<std::endl;
    std::cout << Form("[INFO] 2NLL + c = %.2f",  minNll) << std::endl;
    std::cout << "-----------------------" << std::endl;

    if (!silent) {
      /*
      std::cout << "After Minimization ------------------  " <<std::endl;
      std::cout << bkg->getCurrentPdf()->GetName() << " " << minNll <<std::endl;
      bkg->Print("v");
      bkg->getCurrentPdf()->getParameters(*data)->Print("V");
      std::cout << " ------------------------------------  " << std::endl;
  
      params->Print("V");
      */
      std::cout << "[INFO] AFTER FITTING" << std::endl;
      std::cout << "[INFO] Function was " << bkg->getCurrentPdf()->GetName() <<std::endl;
      std::cout << "[INFO] Correction Applied (NLL units, 0.5 per parameter) is " << bkg->getCorrection() <<std::endl;
      std::cout << "[INFO] 2NLL + c = " <<  minNll << std::endl;
      std::cout << "-----------------------" << std::endl;
    }
      
    if (minNll < global_minNll){
      global_minNll = minNll;
      snap.assignValueOnly(*params);
      best_index=id;
    }
  } // end loop over pdf_index 
  cat->setIndex(best_index);
  params->assignValueOnly(snap);

  std::cout << "[INFO] Best fit Function -- " << bkg->getCurrentPdf()->GetName() << " " << cat->getIndex() <<std::endl;
  //std::cout << "[INFO] Best fit parameters " << std::endl;
  //params->Print("V");
  
  return best_index;
}

  // Novosibirsk pdf with starting values taken from the sidebands (outside [blind_low, blind_high]).
  // It has its own peak + asymmetric tail, so it models the low-mass turn-on by itself (no turn-on is added).
  // RooNovosibirsk convention: tail < 0 gives a tail towards high mass, tail > 0 towards low mass.
  // Parameters are named <ext>_novo_*; the pdf is named pdfName (default: ext).
  RooAbsPdf* makeNovosibirsk(RooRealVar* mass, RooDataHist* h, const char* ext, const char* pdfName = nullptr) {
    double xmin = mass->getMin();
    double xmax = mass->getMax();
    double peak0 = 0.5*(xmin + xmax);
    double width0 = (xmax - xmin)/4.;

    if (h) {
      double sumW = 0., sumWX = 0., sumWX2 = 0., maxW = -1.;
      double peakBin = peak0;
      for (int i = 0; i < h->numEntries(); i++) {
        const RooArgSet* point = h->get(i);
        double x = ((RooRealVar*)point->find(mass->GetName()))->getVal();
        if (x > blind_low && x < blind_high) continue;
        double w = h->weight();
        sumW += w; sumWX += w*x; sumWX2 += w*x*x;
        if (w > maxW) { maxW = w; peakBin = x; }
      }
      if (sumW > 0.) {
        double mean = sumWX/sumW;
        double var = sumWX2/sumW - mean*mean;
        peak0 = peakBin;
        if (var > 0.) width0 = 0.5*sqrt(var);
      }
    }
    double peakHigh = std::min(blind_low, xmax);
    peak0 = std::max(xmin, std::min(peak0, peakHigh));
    width0 = std::max(5.5, std::min(width0, xmax - xmin));

    RooRealVar *peak  = new RooRealVar(Form("%s_novo_peak", ext),  "Novosibirsk peak",  peak0,  xmin, peakHigh);
    RooRealVar *width = new RooRealVar(Form("%s_novo_width", ext), "Novosibirsk width", width0, 5., xmax - xmin);
    // The tail is parametrised as tail = -f * tauMax(peak, width), with f in [0, 1] the floating parameter and
    //   tauMax = min(3, asinh(width*sqrt(ln4)/(peak - xmin))/sqrt(ln4)).
    // The Novosibirsk has a hard edge on the low-mass side at x_edge = peak - width*sqrt(ln4)/sinh(|tail|*sqrt(ln4)),
    // below which the pdf is 0. |tail| <= tauMax is exactly x_edge <= xmin: the edge never enters the fit range
    // (a kink in the NLL there spoils the Hessian). Only tail <= 0 (tail towards high mass) is allowed.
    RooRealVar *tailFrac = new RooRealVar(Form("%s_novo_tailfrac", ext), "Novosibirsk tail fraction", 0.5, 0., 1.);
    RooFormulaVar *tail = new RooFormulaVar(Form("%s_novo_tail", ext), "Novosibirsk tail",
        Form("-@0*TMath::Min(3.0, TMath::ASinH(@1*sqrt(log(4.0))/TMath::Max(@2-%.6f, 1e-3))/sqrt(log(4.0)))", xmin),
        RooArgList(*tailFrac, *width, *peak));
    std::cout << "[INFO] Novosibirsk start values: peak=" << peak0 << " width=" << width0 << " tail=" << tail->getVal() << std::endl;
    return new RooNovosibirsk(pdfName ? pdfName : ext, "Novosibirsk", *mass, *peak, *width, *tail);
  }

  // Novosibirsk x PolExpo(order) = Novosibirsk * exp(c_1 x + ... + c_order x^order), x = (mass-150)/90.
  // order 0 is the plain Novosibirsk (baseline of the F-test ladder). The exponent has no constant term, and all
  // c_i start at 0, so every order starts from the plain Novosibirsk shape; the pdf is positive by construction.
  RooAbsPdf* makeNovoPolExpo(PdfModelBuilder &pdfsModel, RooRealVar* mass, RooDataHist* h, const char* ext, int order) {
    if (order == 0) return makeNovosibirsk(mass, h, ext);

    // PdfModelBuilder::getPolExpo (called directly: getPdf returns NULL for PolExpo order 1)
    RooAbsPdf *polExpo = pdfsModel.getPolExpo(Form("%s_polexp%d", ext, order), order);
    if (!polExpo) return nullptr;
    RooArgSet *pePars = polExpo->getParameters((const RooArgSet*)0);
    for (int i = 0; i < order; i++) {
      RooRealVar *c = dynamic_cast<RooRealVar*>(pePars->find(Form("%s_polexp%d_c%d", ext, order, i)));
      if (c) c->setVal(0.0);
      else std::cerr << "[WARNING] makeNovoPolExpo: PolExpo parameter c" << i << " not found" << std::endl;
    }
    delete pePars;

    RooAbsPdf *novo = makeNovosibirsk(mass, h, ext, Form("%s_novopdf", ext));
    return new RooProdPdf(ext, "Novosibirsk x PolExpo", RooArgList(*novo, *polExpo));
  }

  // Function to create background PDF with optional turn-on function and Z resonance
  RooAbsPdf* createBackgroundWithTurnOn(PdfModelBuilder &pdfsModel, string type, int order, int catNumber,RooRealVar* nZ_cat_original, RooRealVar* mass, const char* ext, bool includeTurnOn = true, bool includeZ = false, RooWorkspace* dataWS = nullptr, std::string turnOnType = "Fermi") {
  
    // Get the base background PDF from the PdfModelBuilder using the existing getPdf function
    RooAbsPdf *basePdf = nullptr;
    if (type == "NovoPolExpo" || type == "Novosibirsk") {
      RooDataHist* novoHist = dataWS ? dynamic_cast<RooDataHist*>(dataWS->data(Form("rooHist_data_cat%d", catNumber))) : nullptr;
      if (type == "Novosibirsk") basePdf = makeNovosibirsk(mass, novoHist, ext);   // plain Novosibirsk
      else basePdf = makeNovoPolExpo(pdfsModel, mass, novoHist, ext, order);       // Novosibirsk x PolExpo(order)
    }
    else basePdf = getPdf(pdfsModel, type, order, ext);

    
    if (!basePdf) {
      cerr << "[ERROR] Could not create base PDF of type " << type << " order " << order << endl;
      return nullptr;
    }
    
    // ── NEW: freeze bern5 p0 = 0 ──────────────────────────────────────────
    if (type == "Bernstein" && order == 5) {
      RooArgSet* pars = basePdf->getParameters((const RooArgSet*)0);
      TIterator* it = pars->createIterator();
      RooAbsArg* a;
      while ((a = (RooAbsArg*)it->Next())) {
        TString n = a->GetName();
        // Parameter names from PdfModelBuilder follow the pattern ext_bern5_p0
        if (n.Contains("bern5") && n.Contains("p0")) {
          RooRealVar* p0 = dynamic_cast<RooRealVar*>(a);
          if (p0) {
            p0->setVal(0.0);
            p0->setConstant(kTRUE);
            std::cout << "[INFO] Frozen " << p0->GetName() << " = 0 (fixed)" << std::endl;
          }
        }
      }
      delete it;
      delete pars;
    }
    // ── freeze PolExpo4 c0 = 0 to remove c0/c2 degeneracy ────────────
    //if (type == "PolExpo" && order >= 0) {
    //  RooArgSet* pars = basePdf->getParameters((const RooArgSet*)0);
    //  TIterator* it = pars->createIterator();
    //  RooAbsArg* a;
    //  while ((a = (RooAbsArg*)it->Next())) {
    //    TString n = a->GetName();
    //    if (n.Contains("_c0")) {
    //      RooRealVar* c0 = dynamic_cast<RooRealVar*>(a);
    //      if (c0) {
    //        c0->setVal(0.0);
    //        c0->setConstant(kTRUE);
    //        std::cout << "[INFO] Frozen " << c0->GetName() << " = 0 (PolExpo4 c0/c2 degeneracy fix)" << std::endl;
    //      }
    //    }
    //  }
    //  delete it;
    //  delete pars;
    //}

    // Fix normalization for RooAddPdf if it's one
    if (basePdf->InheritsFrom("RooAddPdf")) {
      RooArgSet normSet(*mass);
      ((RooAddPdf*)basePdf)->fixCoefNormalization(normSet);
    }
    
    RooAbsPdf *currentPdf = basePdf;
    
    // Novosibirsk already models the turn-on: never multiply it by a turn-on
    const bool isNovo = (type == "NovoPolExpo" || type == "Novosibirsk");
    if (includeTurnOn && isNovo) {
      std::cout << "[INFO] " << type << ": turn-on not added (already modelled by the Novosibirsk)" << std::endl;
    }

    // Add turn-on function if requested
    if (includeTurnOn && !isNovo) {  // RE-ENABLE turn-on
      // Create turn-on function (Fermi-Dirac) - constrained parameters for stability
      // Turn-on function: 1/(1 + exp((cutoff - x)/beta))
      // Tight ranges around physically motivated values

      //cout << "[INFO] Adding turn-on function for " << type << " order " << order << endl;
      
      // Alternative turn-on functions to try:
      RooGenericPdf *turnOnPdf = nullptr;
      
      // Option 1: Fermi-Dirac (current default)
      RooRealVar *cutoff = new RooRealVar(Form("%s_turnon_cutoff", ext), "Turn-on cutoff", 45, 30,80 );
      RooRealVar *beta = new RooRealVar(Form("%s_turnon_beta", ext), "Turn-on beta", 8, 1, 15);
      RooRealVar *alpha = new RooRealVar(Form("%s_turnon_alpha", ext),"Threshold exponent",0.2, 0.08, 3.0);
      if (turnOnType == "Fermi") {
        turnOnPdf = new RooGenericPdf(Form("%s_turnon", ext), "Fermi-Dirac turn-on", 
                                    "1.0/(1.0 + TMath::Exp((@1-@0)/@2))", 
                                    RooArgList(*mass, *cutoff, *beta));
      }
      else if (turnOnType == "Alternative") {
        //RooRealVar *cutoff = new RooRealVar(Form("%s_turnon_cutoff", ext),"Threshold",45, 30, 80);
        turnOnPdf = new RooGenericPdf(Form("%s_turnon", ext), "Sharp turn-on", 
                                    "pow(@0-@1,@2)", 
                                    RooArgList(*mass, *cutoff, *alpha));
      }
      // Option 2: Error Function turn-on (smoother)
      else if (turnOnType == "Erf" || type == "PowerLaw" || type == "PowerLawSingle") {
        turnOnPdf = new RooGenericPdf(Form("%s_turnon", ext), "Error function turn-on", 
                                    "0.5*(1.0 + TMath::Erf((@0-@1)/@2))", 
                                    RooArgList(*mass, *cutoff, *beta));
      }
      // Option 3: Double exponential (sharp cut)
      else if (turnOnType == "DExp" || type == "Exponential") {
        turnOnPdf = new RooGenericPdf(Form("%s_turnon", ext), "Double exponential turn-on", 
                                    "1.0 - TMath::Exp(-TMath::Exp((@0-@1)/@2))", 
                                    RooArgList(*mass, *cutoff, *beta));
      }
      // Default: Fermi-Dirac
      else {
        turnOnPdf = new RooGenericPdf(Form("%s_turnon", ext), "Turn-on function", 
                                    "1.0/(1.0 + TMath::Exp((@1-@0)/@2))", 
                                    RooArgList(*mass, *cutoff, *beta));
      }
      
      currentPdf = new RooProdPdf(Form("_%s_%s_with_turnon",type.c_str(), ext), 
                                "Background with turn-on", 
                                RooArgList(*currentPdf, *turnOnPdf));
    }
    
    // Add Z resonance if requested
    if (includeZ && dataWS) {  // RE-ENABLE Z with FIXED approach
      RooAbsPdf *zModel = dataWS->pdf(Form("model_Z_c%d", catNumber));
      RooDataHist* data_hist = dynamic_cast<RooDataHist*>(dataWS->data(Form("rooHist_data_cat%d", catNumber)));
      if (!data_hist) {
          std::cerr << "Error: dataset is not a RooDataHist!" << std::endl;
      }
      
      if (zModel) {
        //cout << "[INFO] Adding Z resonance model for " << type << " order " << order << endl;
        
        // Freeze Z shape parameters but release some key ones for fitting
        RooArgSet *zParams = zModel->getParameters(RooArgSet(*mass));
        TIterator *iter = zParams->createIterator();
        RooRealVar *param;
        while ((param = (RooRealVar*)iter->Next())) {
          // Release Z mass and width for better fitting
            param->setConstant(kTRUE);
            //cout << "[INFO] Freezing Z parameter: " << param->GetName() << " = " << param->getVal() << " (fixed)" << endl;
          }
        
        delete iter;
        delete zParams;
        
        // Debug: Check Z PDF normalization
        RooArgSet normSet(*mass);
        //double zIntegral = zModel->createIntegral(normSet)->getVal();
        //cout << "[DEBUG] Z PDF integral over mass range: " << zIntegral << endl;
        
        // Set up Z normalization - INDEPENDENT NORMALIZATION approach
        
        //cout << "[INFO] Adding Z resonance using independent normalizations" << endl;
        

        //nZ_cat_original is the fraction wrt to dataentries
        RooRealVar *z_norm = new RooRealVar(Form("%s_z_norm", ext), "Z normalization", 
                                          nZ_cat_original->getVal(), nZ_cat_original->getVal()*0.1 , nZ_cat_original->getVal()*10 ); 
        
        //cout << "[INFO] Using independent normalizations:" << endl;
        //cout << "  Background norm: " << bkg_norm->getVal() << " events" << endl;
        //cout << "  Z norm: " << z_norm->getVal() << "bkg_norm events" << endl;
        
        // Combine with independent normalizations (no coefficients!)
        currentPdf = new RooAddPdf(Form("%s_with_z", ext),
                                  "Background with Z",
                                  RooArgList(*zModel, *currentPdf),
                                  RooArgList(*z_norm));
        
        //cout << "[INFO] Z added with independent normalizations" << endl;
      } else {
        cout << "[WARNING] Could not find Z model 'model_Z_c2' in workspace" << endl;
      }
    }
    
    return currentPdf;
}

int main(int argc, char* argv[]){
  
  setTDRStyle();
  writeExtraText = true;       // if extra text
  extraText  = "Preliminary";  // default extra text is "Preliminary"
  lumi_8TeV  = "19.1 fb^{-1}"; // default is "19.7 fb^{-1}"
  lumi_7TeV  = "4.9 fb^{-1}";  // default is "5.1 fb^{-1}"
  lumi_sqrtS = "13 TeV";       // used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
  string year_ = "2016";
  //int year_ = 2017;

  string fileName;
  // Default to your specific workspace
  //string workspaceFile = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/workspace_step2.root";
  //string workspaceFile = " /t3home/ratramon/ggHBB/Btob_studies/bb_analysis/scripts/workspace.root";
  int ncats;
  int singleCategory;
  int catOffset;
  string datfile;
  string outDir;
  string outfilename;
  bool verbose=false;
  bool saveMultiPdf=false;
  bool includeTurnOn=false;  // Disable turn-on function by default for speed
  bool includeZ=false;      // Option to include Z resonance
  bool blindSignalRegion=false; // Option to blind signal region [100-140] GeV in fit
  int isFlashgg_ =0;
  int catNumber=2;
  string flashggCatsStr_;
  vector<string> flashggCats_;
  bool isData_ =0;
  int maxIt=6;
  bool iterativeFit = false;
  std::string turnOnType = "Fermi"; // Default

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h",                                                                                  "Show help")
    ("infilename,i", po::value<string>(&fileName),                                              "In file name")
    //("workspace,w", po::value<string>(&workspaceFile)->default_value("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/workspace_step2.root"), "Workspace file with RooDataHist and Z model")
    ("includeTurnOn", po::value<bool>(&includeTurnOn)->default_value(false),                    "Include turn-on function for low mass threshold")
    ("includeZ", po::value<bool>(&includeZ)->default_value(false),                             "Include Z resonance model from workspace")
    ("blindSignalRegion", po::value<bool>(&blindSignalRegion)->default_value(true),           "Blind signal region [100-140] GeV in fit (not just plots)")
    ("maxIt", po::value<int>(&maxIt)->default_value(6),           "Maximum number of iterations for the fit")
    ("ncats,c", po::value<int>(&ncats)->default_value(5),                                       "Number of categories")
    ("singleCat", po::value<int>(&singleCategory)->default_value(0),                           "Run A single Category (default: category 0)")
    ("datfile,d", po::value<string>(&datfile)->default_value("dat/fTest.dat"),                  "Right results to datfile for BiasStudy")
    ("outDir,D", po::value<string>(&outDir)->default_value("plots/fTest"),                      "Out directory for plots")
    ("saveMultiPdf", po::value<string>(&outfilename),                                           "Save a MultiPdf model with the appropriate pdfs")
    ("catNumber", po::value<int>(&catNumber)->default_value(2),                                       "Number of the category")
    ("runFtestCheckWithToys",                                                                   "When running the F-test, use toys to calculate pvals (and make plots) ")
    ("is2012",                                                                                  "Run 2012 config")
    ("isFlashgg",  po::value<int>(&isFlashgg_)->default_value(1),                               "Use Flashgg output ")
    ("isData",  po::value<bool>(&isData_)->default_value(0),                                    "Use Data not MC ")
    ("flashggCats,f", po::value<string>(&flashggCatsStr_)->default_value("UntaggedTag_0,UntaggedTag_1,UntaggedTag_2,UntaggedTag_3,UntaggedTag_4,VBFTag_0,VBFTag_1,VBFTag_2,TTHHadronicTag,TTHLeptonicTag,VHHadronicTag,VHTightTag,VHLooseTag,VHEtTag"),                  "Flashgg category names to consider")
    ("year", po::value<string>(&year_)->default_value("2016"),                                  "Dataset year")
    ("catOffset", po::value<int>(&catOffset)->default_value(0),                                 "Category numbering scheme offset")
    ("verbose,v",                                                                               "Run with more output")
    ("iterativeFit", po::value<bool>(&iterativeFit)->default_value(false), "Enable iterative fitting: first fit [mass->getMin(),78]+>140 w/o Z, then [mass->getMin(),100]+>140 with Z")
    ("turnOnType", po::value<std::string>(&turnOnType)->default_value("Fermi"), "Type of turn-on function: Fermi, Erf, DExp")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc,argv,desc),vm);
  po::notify(vm);
  if (vm.count("help")) { cout << desc << endl; exit(1); }
  saveMultiPdf = vm.count("saveMultiPdf");

  if (vm.count("verbose")) verbose=true;
  if (vm.count("runFtestCheckWithToys")) runFtestCheckWithToys=true;




  if (!verbose) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
    gErrorIgnoreLevel=kWarning;
  }
  
  // Speed up numerical integrations globally
 // RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");
 // RooAbsReal::defaultIntegratorConfig()->getConfigSection("RooAdaptiveGaussKronrodIntegrator1D").setRealValue("maxSeg", 100);
 // RooAbsReal::defaultIntegratorConfig()->getConfigSection("RooAdaptiveGaussKronrodIntegrator1D").setRealValue("epsAbs", 1e-6);
 // RooAbsReal::defaultIntegratorConfig()->getConfigSection("RooAdaptiveGaussKronrodIntegrator1D").setRealValue("epsRel", 1e-6);
  
  
  // DO NOT override ncats for flashgg - keep single category behavior
  // if (isFlashgg_==1){
  //   ncats= flashggCats_.size();
  // }
  


  if(verbose) std::cout << "[INFO] SaveMultiPdf? " << saveMultiPdf << std::endl;
  TFile *outputfile;
  RooWorkspace *outputws;

  if (saveMultiPdf){
  outputfile = new TFile(outfilename.c_str(),"RECREATE");
  outputws = new RooWorkspace(); outputws->SetName("multipdf");
  }

  system(Form("mkdir -p %s",outDir.c_str()));
  
  // Load the main data file (if provided)
  //TFile *inFile = nullptr;
  //RooWorkspace *inWS = nullptr;
  
  //if (!fileName.empty()) {
  //  inFile = TFile::Open(fileName.c_str());
  //  std::cout<< "Loading file: " << fileName << std::endl;
  //  if(isFlashgg_){
  //    if (isData_){
  //      inWS = (RooWorkspace*)inFile->Get("wS");
  //    } else {
  //      inWS = (RooWorkspace*)inFile->Get("wS");
  //    }
  //  } else {
  //    inWS = (RooWorkspace*)inFile->Get("wS");
  //  }
  //}
  
  // Load the workspace with RooDataHist and Z model
  TFile *wsFile = TFile::Open(fileName.c_str());
  if (saveMultiPdf){
    transferMacros(wsFile,outputfile);

    RooRealVar *intL; 
    RooRealVar *sqrts;

      //intL  = (RooRealVar*)inWS->var("IntLumi");
    intL  = intLumi_;
    sqrts = new RooRealVar("SqrtS","SqrtS",13);
    outputws->import(*intL);
    outputws->import(*sqrts);
    std::cout << "[INFO] got intL and sqrts " << intL << ", " << sqrts << std::endl;
  }

  if (!wsFile || wsFile->IsZombie()) {
    cerr << "[ERROR] Cannot open workspace file: " << fileName << endl;
    return 1;
  }
  
  RooWorkspace *dataWS = (RooWorkspace*)wsFile->Get("WS");
  RooRealVar *nZ_cat = dataWS->var(Form("nZ_cat%d", catNumber));
  RooDataHist  *hist_data       = (RooDataHist*)dataWS->data(Form("rooHist_data_cat%d", catNumber));
  //RooRealVar* nZ_cat_original = new RooRealVar(*nZ_cat);
  RooRealVar* nZ_cat_original = new RooRealVar("nZ_cat_original", "nZ_cat_original", nZ_cat->getVal()/hist_data->sumEntries());
  std::cout << "nZ_cat_original is "<< nZ_cat_original->getVal() << std::endl;

  //RooWorkspace *dataWS = (RooWorkspace*)wsFile->Get("w");
  if (!dataWS) {
    // Try alternative workspace names
    dataWS = (RooWorkspace*)wsFile->Get("cms_hgg_workspace");
    if (!dataWS) {
      dataWS = (RooWorkspace*)wsFile->Get("WS");
      if (!dataWS) {
        dataWS = (RooWorkspace*)wsFile->Get("workspace");
        if (!dataWS) {
          cerr << "[ERROR] Cannot find workspace in " << fileName << endl;
          wsFile->ls();
          return 1;
        }
      }
    }
  }
  
  std::cout << "[INFO] Loaded workspace from: " << fileName << std::endl;
  if (verbose) {
    std::cout << "[INFO] Workspace contents:" << std::endl;
    dataWS->Print();
  }
  
  // Use dataWS for RooDataHist and inWS for other things (if available)
  //if (!inWS) {
  //  inWS = dataWS;  // Use dataWS as fallback
  //}
  //if (verbose){
  // std::cout << "[INFO]  inWS open " << inWS << std::endl;
  // inWS->Print();
  //}


  // Set up which families of functions you want to test
  vector<string> functionClasses;
  // The two Novosibirsk families model the turn-on by themselves: never combined with the turn-on.
  // Comment/uncomment each line independently. If both are enabled, NovoPolExpo order 0 (= plain Novosibirsk,
  // used only as the baseline of its F-test) is not added to the envelope, since "Novosibirsk" already provides it.
  //functionClasses.push_back("NovoPolExpo");         // Novosibirsk x PolExpo(n), n = 1, 2, ... (order 0 = baseline)
  functionClasses.push_back("Novosibirsk");         // plain Novosibirsk (3 parameters), single order
  //functionClasses.push_back("Laurent");         // Removed: similar to polynomials
  // functionClasses.push_back("KeysPdf");      // Needs special setup with setKeysPdfAttributes
  //functionClasses.push_back("Chebychev");
  functionClasses.push_back("Bernstein");
  //functionClasses.push_back("PEBern");
  functionClasses.push_back("PolExpo");
  functionClasses.push_back("Exponential");
  functionClasses.push_back("PowerLaw");         // Regular PowerLaw (not Generic)
  //functionClasses.push_back("Exponential");
  // functionClasses.push_back("ExponentialSum");  // Removed: doesn't fit well
  // functionClasses.push_back("Chebychev");       // Removed
 // functionClasses.push_back("PowerLawSingle");   // Alternative PowerLaw implementation
  // functionClasses.push_back("PowerLawGeneric"); // Replaced with regular PowerLaw
  // functionClasses.push_back("PowerLawSum");     // Removed: doesn't fit well
  //  functionClasses.push_back("Polynomial");
  //map<string,string> namingMap;
  //namingMap.insert(pair<string,string>("Bernstein","pol"));
  //namingMap.insert(pair<string,string>("Exponential","exp"));
  //namingMap.insert(pair<string,string>("ExponentialSum","expsum"));
  //namingMap.insert(pair<string,string>("PolExpo","polexp"));
  //namingMap.insert(pair<string,string>("Chebychev","che"));
   //namingMap.insert(pair<string,string>("Laurent","lau"));
  // namingMap.insert(pair<string,string>("PowerLawSum","pow"));
  //  namingMap.insert(pair<string,string>("Chebychev","che"));
 // namingMap.insert(pair<string,string>("Polynomial","pol"));

  FILE *resFile ;
  //if  (singleCategory >-1) resFile = fopen(Form("%s/fTestResults_%s.txt",outDir.c_str(),flashggCats_[singleCategory].c_str()),"w");
  resFile = fopen(Form("%s/fTestResults.txt",outDir.c_str()),"w");
  vector<map<string,int> > choices_vec;
  vector<map<string,std::vector<int> > > choices_envelope_vec;
  vector<map<string,RooAbsPdf*> > pdfs_vec;

  PdfModelBuilder pdfsModel;
  RooRealVar *mass = (RooRealVar*)dataWS->var(Form("dijet_mass_c%d",catNumber)); 
  //RooRealVar *mass = (RooRealVar*)dataWS->var("mjj"); 
  //std:: cout << "[INFO] Got mass from ws " << mass << std::endl;
  //mass->Print("v");
  std:: cout << "[INFO] Mass range: [" << mass->getMin() << ", " << mass->getMax() << "]" << std::endl;
  
  // Define blinding ranges for fit and plot
  mass->setRange("blind_low", mass->getMin(), blind_low);   // Sideband basso
  mass->setRange("blind_high", blind_high, mass->getMax());   // Sideband alto
  mass->setRange("blindFit_low", mass->getMin(), blindFit_low);   // Sideband basso
  mass->setRange("blindFit_high", blindFit_high, mass->getMax());   // Sideband alto
  if (blindSignalRegion) {
    std::cout << Form("[INFO] Signal region [%d-%d] GeV will be blinded in fit",(int)blind_low,(int)blind_high) << std::endl;
  }
  
  pdfsModel.setObsVar(mass);
  double upperEnvThreshold = 0.01; // upper threshold on prob_ftest to include function in envelope (looser than truth function)
  // penalize small improvements in NLL
  double minGofThreshold = 0.05;  // minimal goodness of fit to include function in envelope

  fprintf(resFile,"Truth Model & d.o.f & $\\Delta NLL_{N+1}$ & $p(\\chi^{2}>\\chi^{2}_{(N\\rightarrow N+1)})$ \\\\\n");
  fprintf(resFile,"\\hline\n");

  std::string ext = Form("%s_13TeV",year_.c_str());

  std::cout << "[INFO] Number of categories to process: " << ncats << std::endl;
  // Start processing categories
  for (int cat=0; cat<ncats; cat++){

    map<string,int> choices;
    map<string,std::vector<int> > choices_envelope;
    map<string,RooAbsPdf*> pdfs;
    string catname = Form("cat%d",cat);
    

    // Option 2 (equivalente): Usa come input un RooDataHist già binned
    string data_name =Form("rooHist_data_cat%d", catNumber);
    RooDataHist  *data_in       = (RooDataHist*)dataWS->data(data_name.c_str());
    int nBinsNotBlinded = 0;
    int nBins =0;
    if (!data_in) {
      cerr << "[ERROR] Could not find dataset '" << data_name << "' in workspace" << endl;
      continue; // Salta questa categoria e prova con la successiva
    } else {
    // Get the first observable (e.g. "mass")
    RooRealVar* obs = (RooRealVar*)data_in->get()->first();

    if (!obs) {
        cerr << "[ERROR] Could not retrieve observable from dataset " << data_name << endl;
    } else {
    int nbins = obs->getBins();
    double xmin = obs->getMin();
    double xmax = obs->getMax();

    cout << "[INFO] Dataset " << data_name
         << " has binning: " << nbins
         << " bins from " << xmin << " to " << xmax << endl;



    double low = xmin;
    double high = xmin + (xmax-xmin)/nbins;
    double center = 0.5 * (low + high);
    for (int i = 0; i < nbins; ++i) {
        //std::cout<<Form("Low : %d High : %d Center : %d    ", (int)low, (int)high, (int)center);
        
        if (center <= blind_low || center >= blind_high) {
            ++nBinsNotBlinded;
            
        }
        else{

        }
        low  = low + (xmax-xmin)/nbins;
        high = high + (xmax-xmin)/nbins;
        center = 0.5 * (low + high);
    }

    cout << "[INFO] nBinsNotBlinded = " << nBinsNotBlinded << endl;
    cout << "[INFO] nBins = " << nbins << endl;
    }
    }
    
    // Ottieni il binning dalla variabile di massa nella workspace (che dovrebbe corrispondere al RooDataHist)

    //cout << "[INFO] Using binning from workspace mass variable: " << nBinsForFit << " bins" << endl;
    
    std::cout << "entries dataset " << data_in->sumEntries() << std::endl;
    
    // Debug: Stampa alcune informazioni sul RooDataHist originale
    //cout << "[DEBUG] Original RooDataHist info:" << endl;
    //data_in->Print("v");
    
    // Implementa il rebinning se richiesto
    RooAbsData *data;

    data = data_in;
    // Debug: Stampa alcune informazioni sui dati che useremo
    //cout << "[DEBUG] Using data type: " << data->ClassName() << endl;
    
  


    
    //RooArgList storedShapePdfs("storeShape");
    RooArgList storedPdfs("store");

    fprintf(resFile,"\\multicolumn{4}{|c|}{\\textbf{Category %d}} \\\\\n",cat);
    fprintf(resFile,"\\hline\n");

    double MinimimNLLSoFar=1e10;
    int simplebestFitPdfIndex = 0;

    // definition of struct
    struct PdfCandidate {
      RooAbsPdf* pdf;
      int order;
      double gofProb;
      double twice_nll;
      int nfloat;                  // <-- add this
      bool supersededByNext = false;
    };

    // Standard F-Test to find the truth functions
    for (vector<string>::iterator funcType=functionClasses.begin(); funcType!=functionClasses.end(); funcType++){

      // Enable turn-on for all functions
      bool currentIncludeTurnOn = includeTurnOn;
      currentIncludeTurnOn = includeTurnOn && (*funcType != "NovoPolExpo") && (*funcType != "Novosibirsk");   // Novosibirsk already models the turn-on
  
      std::cout << "\n\n======================================= " << std::endl;
      std::cout << "====> FAMILY " << funcType->c_str() << std::endl;
      std::cout << "======================================= \n\n" << std::endl;


      double thisNll=0.; double prevNll=0.; double chi2_Ftest=0.; double prob_Ftest=0; 
      // NovoPolExpo starts at order 0 (plain Novosibirsk, baseline of the nested F-test); all other families at order 1
      const int firstOrder = (*funcType == "NovoPolExpo") ? 0 : 1;
      int order=firstOrder; int prev_order=0; int cache_order=0;

      //RooAbsPdf *prev_pdf=NULL;
      //RooAbsPdf *cache_pdf=NULL;
      std::vector<int> pdforders;


      // Now run loop to determine functions inside envelope
      std::cout << "===> F-TEST and GOF for ENVELOPE determination" << std::endl;
      if (saveMultiPdf){
        chi2_Ftest=0.;
        thisNll=0.;
        prevNll=0.;
        prob_Ftest=0.;
        order=firstOrder;
        prev_order=0;
        cache_order=0;
        std::cout << "[INFO] Upper end Threshold for highest order function " << upperEnvThreshold <<std::endl;
        
        
        std::vector<PdfCandidate> candidates;
        std::cout<<"Max iteration is "<<maxIt<<std::endl;
        while (prob_Ftest<upperEnvThreshold && order < maxIt){ 
          if ((*funcType == "PowerLaw" || *funcType == "Novosibirsk") && order > 1) {
            std::cout << "[INFO] Skipping " << *funcType << " order " << order << " (only order 1 allowed)" << std::endl;
            break;
          }
          // HARDCODED: skip Bernstein order 4 (comment out to restore)
          //if (*funcType == "Bernstein" && order == 4) {
          //  std::cout << "[INFO] Skipping Bernstein order " << order << " (hardcoded)" << std::endl;
          //  order++;
          //  continue;
          //}
          // HARDCODED: skip Exponential order 5 (comment out to restore)
          if (*funcType == "Exponential" && order == 5) {
            std::cout << "[INFO] Skipping Exponential order " << order << " (hardcoded)" << std::endl;
            order++;
            continue;
          }
          cout << "==> " << *funcType << " " << order << endl;
          RooAbsReal::defaultIntegratorConfig()->setEpsAbs(1e-9);
          RooAbsReal::defaultIntegratorConfig()->setEpsRel(1e-9);
          RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");
          RooAbsReal::defaultIntegratorConfig()->getConfigSection("RooAdaptiveGaussKronrodIntegrator1D").setRealValue("maxSeg", 1000);
          RooAbsPdf *bkgPdf = createBackgroundWithTurnOn(pdfsModel,*funcType,order,catNumber, nZ_cat_original, mass,Form("env_pdf_%s_%d_cat%d",(*funcType).c_str(),order,catNumber), currentIncludeTurnOn, includeZ, dataWS, turnOnType);

          // Additional check: fix normalization for any RooAddPdf
          if (bkgPdf && bkgPdf->InheritsFrom("RooAddPdf")) {
            RooArgSet normSet(*mass);
            ((RooAddPdf*)bkgPdf)->fixCoefNormalization(normSet);
          }
          
          if (!bkgPdf ){
            // assume this order is not allowed
            if (order >maxIt-1) 
              { std::cout << " [WARNING] could not add ] " << std::endl; break ;}
            else
              {order++;}
          }
          else {
            //the function has been created correctly
            int fitStatus=0;
            double chi2FromFit = -1;
            runFit(bkgPdf, data, &thisNll, &fitStatus, 3, mass, nZ_cat_original, &chi2FromFit, blindSignalRegion, includeTurnOn);

            if (fitStatus!=0) {
              if (order == 1){
                chi2_Ftest=0;
                prob_Ftest=0; // go on with testing
              }
              else{
                std::cout << "[WARNING] Warning -- Fit status for " << bkgPdf->GetName() << " at " << fitStatus <<std::endl;
                chi2_Ftest=0;
                prob_Ftest=0; //go on with testing
                }
              }
            else if (*funcType == "NovoPolExpo" && order == firstOrder) {
              // baseline (plain Novosibirsk): nothing to compare with
              chi2_Ftest = 0.;
              prob_Ftest = 0.;
              }
            else{
              chi2_Ftest = 2.*(prevNll-thisNll);
              prob_Ftest = TMath::Prob(chi2_Ftest,order-prev_order);
              }

            double twice_Nll = 2.*thisNll;
            if (chi2_Ftest<0. && order>firstOrder) {
              chi2_Ftest=0.;
              prob_Ftest=0;
              }
            



            std::cout << "[INFO] function type | order | prevNLL  | thisNLL  | chi2     | prob\n";
            std::cout << "[INFO] "
                      << std::left  << std::setw(14) << *funcType << " | "
                      << std::right << std::setw(5)  << order      << " | "
                      << std::setw(8)  << prevNll    << " | "
                      << std::setw(8)  << thisNll    << " | "
                      << std::setw(8)  << chi2_Ftest << " | "
                      << std::setw(8)  << prob_Ftest << "\n";

            prevNll=thisNll;
            cache_order=prev_order;
            //cache_pdf=prev_pdf;

            // Calculate goodness of fit (will use toys for lowstats)
            double gofProb =0; 
            plot1(mass,bkgPdf,data,nBinsNotBlinded, Form("%s/%s%d_cat%d",outDir.c_str(),funcType->c_str(),order,catNumber),flashggCats_,fitStatus,&gofProb, chi2FromFit, true, nBins);
            int pullsStatus = 1;
            // Also create component plot when Z is included
            if (includeZ && dataWS) {
              RooAbsPdf *zModel = dataWS->pdf(Form("model_Z_c%d", catNumber));
              if (zModel) {
                
                plotComponents(mass,bkgPdf,zModel,data,nBinsNotBlinded, nZ_cat_original, Form("%s/%s%d_cat%d",outDir.c_str(),funcType->c_str(),order,catNumber),flashggCats_,fitStatus,&gofProb,thisNll + bkgPdf->getVariables()->getSize()*0.5, &pullsStatus, chi2FromFit,blindSignalRegion, nBins);
              }
            }
            std::cout<<Form("[INFO] Prob. Ftest is %.3f | Max Value is %.3f ",prob_Ftest, upperEnvThreshold)<<std::endl;
            std::cout<<Form("[INFO] Gof Prob is %.3f | Min Value is %.3f ",gofProb, minGofThreshold)<<std::endl;
            std::cout<<Form("[INFO] Fit Status is %d ",fitStatus)<<std::endl;
            std::cout<<Form("[INFO] pullsStatus Status is %d ",pullsStatus)<<std::endl;
            if (prob_Ftest < upperEnvThreshold) { // Looser requirements for the envelope


              if ((gofProb > minGofThreshold) && (fitStatus==0) && (pullsStatus==0)) { // minimal requirement on the goodness of fit

                int nfloat = 0;
                RooArgSet* candPars = bkgPdf->getParameters(*data);
                for (auto p : *candPars) {
                    RooRealVar* rv = dynamic_cast<RooRealVar*>(p);
                    if (rv && !rv->isConstant()) nfloat++;
                }
                std::cout << "[INFO] Adding to Envelope " << bkgPdf->GetName() << funcType->c_str()<<" "<< gofProb 
                  << Form(" correction c = %d", nfloat)
                  << Form(" order o = %d", order)
                  << Form(" 2xNLL + c is %.2f", twice_Nll + nfloat) <<  std::endl;
                // here new code starts
                //std::cout<<"Hello I am trying the pdf as a shapebinpdf "<<std::endl; 
                //RooDataHist* rdh = dynamic_cast<RooDataHist*>(data);
                //TH1* histo = rdh->createHistogram("massHist", *mass);
                //RooArgSet* paramsSet = bkgPdf->getParameters(*data);
                //RooArgList paramsList(*paramsSet);
                //RooParametricShapeBinPdf* shapePdf = new RooParametricShapeBinPdf(
                //    "shapePdf","Parametric binned version of epdf",
                //    *bkgPdf,*mass,paramsList,*histo
                //);
                //storedShapePdfs.add(*shapePdf);
                // here ends new code
                //old code here
                // Mark previous candidate as superseded if this order is a significant improvement
                if (!candidates.empty() && order > firstOrder) {
                  candidates.back().supersededByNext = true;
                  std::cout << "[INFO] Order " << prev_order
                            << " will be superseded by order " << order
                            << " (prob_Ftest=" << prob_Ftest << ")" << std::endl;
                }
                delete candPars;
                candidates.push_back({bkgPdf, order, gofProb, twice_Nll, nfloat, false});

              }
              prev_order=order;
              order++;
            }
            else{
              std::cout<<Form("[INFO] The function %s was NOT added in the envelope since no significant improvement has been found...", (*funcType).c_str())<<std::endl;
              prob_Ftest =0;
              prev_order=order;
              order++;
            }
            
          }
        } // end while

            // Flush candidates into storedPdfs, skipping those superseded by a better higher order
      for (auto& cand : candidates) {
        if (cand.supersededByNext) {
          std::cout << "[INFO] Skipping " << cand.pdf->GetName()
                    << " order " << cand.order
                    << " from envelope: superseded by significantly better higher order." << std::endl;
          continue;
        }
        // NovoPolExpo order 0 is the plain Novosibirsk: skip it if the "Novosibirsk" family provides it already
        if (*funcType == "NovoPolExpo" && cand.order == 0 &&
            std::find(functionClasses.begin(), functionClasses.end(), std::string("Novosibirsk")) != functionClasses.end()) {
          std::cout << "[INFO] Skipping " << cand.pdf->GetName()
                    << " from envelope: identical to the plain Novosibirsk family." << std::endl;
          continue;
        }
        std::cout << "[INFO] Adding to Envelope " << cand.pdf->GetName()
                  << Form(" correction c = %d", cand.nfloat)
                  << Form(" order o = %d", cand.order)
                  << Form(" 2xNLL + c is %.2f", cand.twice_nll + cand.nfloat)
                  << std::endl;
        storedPdfs.add(*cand.pdf);
        pdforders.push_back(cand.order);
        if ((cand.twice_nll + cand.nfloat) < MinimimNLLSoFar) {
            simplebestFitPdfIndex = storedPdfs.getSize() - 1;
            MinimimNLLSoFar = cand.twice_nll + cand.nfloat;
        }
      }
      candidates.clear();

      fprintf(resFile,"%15s & %d & %5.3f & %5.3f \\\\\n",funcType->c_str(),cache_order+1,chi2_Ftest,prob_Ftest);
      choices_envelope.insert(pair<string,std::vector<int> >(*funcType,pdforders));
      }
    } // end loop over families


    fprintf(resFile,"\\hline\n");
    choices_vec.push_back(choices);
    choices_envelope_vec.push_back(choices_envelope);
    pdfs_vec.push_back(pdfs);

    

    if (saveMultiPdf){
      // Put selectedModels into a MultiPdf
      string catindexname;
      string catname;
      if (isFlashgg_){
        catindexname = Form("pdfindex_%s_%s",std::to_string(catNumber).c_str(),ext.c_str());
        catname = Form("%s",std::to_string(cat).c_str());
      } else {
        catindexname = Form("pdfindex_%d_%s",(catNumber),ext.c_str());
        catname = Form("cat%d",(catNumber));
      }
      RooCategory catIndex(catindexname.c_str(),"c");
      RooRealVar nBackground(Form("CMS_hgg_%s_%s_bkg_norm",catname.c_str(),ext.c_str()),"nbkg",data->sumEntries()*(1-nZ_cat_original->getVal()),0.8*data->sumEntries(),1.2*data->sumEntries());
      RooMultiPdf *multiPdf = new RooMultiPdf(Form("CMS_hgg_%s_%s_bkg",catname.c_str(),ext.c_str()),"all pdfs", catIndex, storedPdfs);
      //RooMultiPdf *shapeMultipdf = new RooMultiPdf(Form("CMS_hgg_%s_%s_bkgshape",catname.c_str(),ext.c_str()),"all shape pdfs", catIndex, storedShapePdfs);
       //nBackground.removeRange(); // bug in roofit will break combine until dev branch brought in
      //double check the best pdf!
      int bestFitPdfIndex = getBestFitFunction(multiPdf,data, &catIndex,mass, nZ_cat_original, !verbose, includeTurnOn, blindSignalRegion);
     
      catIndex.setIndex(bestFitPdfIndex);
      std::cout << "// ------------------------------------------------------------------------- //" <<std::endl; 
      std::cout << "[INFO] Created MultiPdf " << multiPdf->GetName() << ", in Category " << cat << " with a total of " << catIndex.numTypes() << " pdfs"<< std::endl;
      storedPdfs.Print();
      
      std::cout << "[INFO] Best Fit Pdf = " << bestFitPdfIndex << ", " << storedPdfs.at(bestFitPdfIndex)->GetName() << std::endl;
      

      std::cout << "[INFO] Simple check of index "<< simplebestFitPdfIndex <<std::endl;
      std::cout << "// ------------------------------------------------------------------------- //" <<std::endl;

      // mass->setBins(nBinsForFit); // Binning now taken from workspace
      //RooDataHist dataBinned(Form("roohist_data_mass_%s",catname.c_str()),"data",*mass,*dataFull);

      // Save it (also a binned version of the dataset
      outputws->import(*multiPdf);
      outputws->import(nBackground);
      outputws->import(catIndex);
      //outputws->import(dataBinned);
      outputws->import(*data);
      plotMultiPdf(mass,multiPdf,&catIndex,data,Form("%s/multipdf_%s",outDir.c_str(),catname.c_str()),flashggCats_,cat,bestFitPdfIndex);

    } // end if saveMultiPdf

  }// end loop over categories 

  if (saveMultiPdf){
    outputfile->cd();
    outputws->Write();
    outputfile->Close();  
  }
  wsFile->Close();
}