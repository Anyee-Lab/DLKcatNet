% manual change for the substrate usage in the training dataset for example
% xylose usage 
% species: Spathaspora_passalidarum
cd ../../Results/ssGEMs/
load('Spathaspora_passalidarum.mat')
model = addrxnBack(model,model_original,{'r_1272'},{''});
max_growth(strcmp(max_growth(:,2),'D-xylose'),:) = []; % later to use this to simulate ocygen influence towards ethanol production
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')


load('yHMPu5000034709_Kluyveromyces_aestuarii.mat')
model = changeGAM(model,30,0.5);
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('yHMPu5000041693_Debaryomyces_nepalensis.mat')
model = addrxnBack(model,model_original,{'r_1094'},{''});
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('Candida_parapsilosis.mat') % anaerobic growth is way lower compared with aerobic comstrain other byproduct production
growthdata(strcmp(growthdata(:,14),'anaerobic'),:) = [];
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('Candida_glabrata.mat')
growthdata(strcmp(growthdata(:,14),'anaerobic'),:) = [];
growthdata(strcmp(growthdata(:,2),'D-glucose'),:) = [];
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('Debaryomyces_hansenii.mat')
model = changeGAM(model,30,0.5);
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('Meyerozyma_guilliermondii.mat')
model.S(strcmp(model.mets,'s_0796[e]'),startsWith(model.rxns,'r_1134')) = 0;
model.S(strcmp(model.mets,'s_0794[c]'),startsWith(model.rxns,'r_1134')) = 0;
model.S(strcmp(model.mets,'s_0796[e]'),startsWith(model.rxns,'r_1139')) = 0;
model.S(strcmp(model.mets,'s_0794[c]'),startsWith(model.rxns,'r_1139')) = 0;
growthdata(strcmp(growthdata(:,2),'D-glucose'),:) = [];

save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')

load('Scheffersomyces_stipitis.mat')
growthdata(strcmp(growthdata(:,14),'anaerobic'),:) = [];
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')


load('yHMPu5000034625_Pichia_kudriavzevii.mat')
model = addrxnBack(model,model_original,{'r_1074'},{''});
model = addrxnBack(model,model_original,{'r_1024'},{''});
max_growth(strcmp(max_growth(:,14),'anaerobic'),:) = [];
save([strain,'.mat'],'enzymedata','growthrates','max_growth','model','MWdata','Protein_stoichiometry','strain','growthdata','rxn2block')
cd ../../Code/ecGEMconstruction