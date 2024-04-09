
#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TMath.h"
#include <iostream>
#include <fstream>
#include <vector>

double langaufun(double *x, double *par) {

   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation),
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.

      // Numeric constants
      double invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      double mpshift  = -0.22278298;       // Landau maximum location

      // Control constants
      double np = 100.0;      // number of convolution steps
      double sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

      // Variables
      double xx;
      double mpc;
      double fland;
      double sum = 0.0;
      double xlow,xupp;
      double step;
      double i;


      // MP shift correction
      mpc = par[1] - mpshift * par[0];

      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];

      step = (xupp-xlow) / np;

      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);

         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }

      return (par[2] * step * sum * invsq2pi / par[3]);
}



TF1 *langaufit(TH1F *his, double *fitrange, double *startvalues, double *parlimitslo, double *parlimitshi, double *fitparams, double *fiterrors, double *ChiSqr, int *NDF)
{
   // Once again, here are the Landau * Gaussian parameters:
   //   par[0]=Width (scale) parameter of Landau density
   //   par[1]=Most Probable (MP, location) parameter of Landau density
   //   par[2]=Total area (integral -inf to inf, normalization constant)
   //   par[3]=Width (sigma) of convoluted Gaussian function
   //
   // Variables for langaufit call:
   //   his             histogram to fit
   //   fitrange[2]     lo and hi boundaries of fit range
   //   startvalues[4]  reasonable start values for the fit
   //   parlimitslo[4]  lower parameter limits
   //   parlimitshi[4]  upper parameter limits
   //   fitparams[4]    returns the final fit parameters
   //   fiterrors[4]    returns the final fit errors
   //   ChiSqr          returns the chi square
   //   NDF             returns ndf

   int i;
   char FunName[100];

   sprintf(FunName,"Fitfcn_%s",his->GetName());

   TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
   if (ffitold) delete ffitold;

   TF1 *ffit = new TF1(FunName,langaufun,fitrange[0],fitrange[1],4);
   ffit->SetParameters(startvalues);
   ffit->SetParNames("Width","MP","Area","GSigma");

   for (i=0; i<4; i++) {
      ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
   }

   his->Fit(FunName,"RB0");   // fit within specified range, use ParLimits, do not plot

   ffit->GetParameters(fitparams);    // obtain fit parameters
   for (i=0; i<4; i++) {
      fiterrors[i] = ffit->GetParError(i);     // obtain fit parameter errors
   }
   ChiSqr[0] = ffit->GetChisquare();  // obtain chi^2
   NDF[0] = ffit->GetNDF();           // obtain ndf

   return (ffit);              // return fit function

}


int langaupro(double *params, double &maxx, double &FWHM) {

   // Searches for the location (x value) at the maximum of the
   // Landau-Gaussian convolute and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.

   double p,x,fy,fxr,fxl;
   double step;
   double l,lold;
   int i = 0;
   int MAXCALLS = 10000;


   // Search for maximum

   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = langaufun(&x,params);

      if (l < lold)
         step = -step/10;

      p += step;
   }

   if (i == MAXCALLS)
      return (-1);

   maxx = x;

   fy = l/2;


   // Search for right x location of fy

   p = maxx + params[0];
   step = params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);

      if (l > lold)
         step = -step/10;

      p += step;
   }

   if (i == MAXCALLS)
      return (-2);

   fxr = x;


   // Search for left x location of fy

   p = maxx - 0.5 * params[0];
   step = -params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;

   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);

      if (l > lold)
         step = -step/10;

      p += step;
   }

   if (i == MAXCALLS)
      return (-3);


   fxl = x;

   FWHM = fxr - fxl;
   return (0);
}


void charge_fit() {
   // Open file with charge (already with cuts)
   std::string file_dir = "/home/marcello/Desktop/Radboud_not_synchro/Master_Thesis/testbeam-analysis/ROOT Langaus fit/";
   std::string file_name = "charge_data_all_cuts_401_S1_3.csv";
   std::ifstream inputFile(file_dir+file_name);
   
   // Check if the file is opened successfully
   if (!inputFile.is_open()) {
      std::cerr << "Error: Unable to open the file data.txt\n";
      return; // Exit with error
   }

   std::vector<double> ones;
   std::vector<double> charge;
   double value;
   
   while (inputFile >> value) charge.push_back(value);
   inputFile.close();

   int size = charge.size();
   std::cout << "size: " << size << std::endl;

   for (int i=0; i<size; i++) ones.push_back(1);

   int n_bins = 500;
   TH1F *hSNR = new TH1F("charge fit","Charge with Langau fit",n_bins,-5,100);

   hSNR->FillN(size,charge.data(),ones.data());

   // Fitting SNR histo
   printf("Fitting...\n");

   // Setting fit range and start values
   double fr[2];
   double sv[4], pllo[4], plhi[4], fp[4], fpe[4];
   fr[0] = 4;
   fr[1] = 100;
   // fr[0] = 0.35*hSNR->GetMean();
   // fr[1] = 3.0*hSNR->GetMean();

   pllo[0]=0.5; pllo[1]=2.0; pllo[2]=1.0; pllo[3]=0.4;
   plhi[0]=5.0; plhi[1]=50.0; plhi[2]=1000000.0; plhi[3]=5.0;
   sv[0]=1.8; sv[1]=20.0; sv[2]=50000.0; sv[3]=3.0;

   double chisqr;
   int    ndf;
   TF1 *fitsnr = langaufit(hSNR,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

   double SNRPeak, SNRFWHM;
   langaupro(fp,SNRPeak,SNRFWHM);

   printf("Fitting done\nPlotting results...\n");
   TCanvas* c1 = new TCanvas("c","c",800,600);

   // Global style settings
   gStyle->SetOptStat(1111);
   gStyle->SetOptFit(111);
   gStyle->SetLabelSize(0.03,"x");
   gStyle->SetLabelSize(0.03,"y");

   // hSNR->GetXaxis()->SetRange(0,200);
   hSNR->Draw("HIST");
   fitsnr->Draw("lsame");

   // legend->AddEntry((TObject*)0, "Some text", "");
   std::string save_name = file_name + "_Charge_fit_ROOT_log.svg";
   c1->SetLogy();
   c1->SaveAs(save_name.data());

   return ;
}

