#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <regex>
//#include <xgboost/c_api.h>

 
using namespace std;
map<string, vector<int>> dic;
vector<string> cat_word;

void init_dic(){

}

int lookup(string comp){
    auto it = find(cat_word.begin(), cat_word.end(), comp);
  
    // If element was found
    if (it != cat_word.end()) 
    {
        int index = it - cat_word.begin();
        return index;
    }
    else
    {
        // If the element is not
        // present in the vector
        return -1;
    }
}


void read_csv()
{
    std::ifstream infile("/home/ishan/workbox/ml_mbti/mbti_1.csv");
    std::string line;
    std::string dels = ".,!\"'*!@#$%^&*{}?;:/-()";
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string type = line.substr(0, 4);
        std::string allfound = line.substr(5, line.length());

        //delete unwanted chars
         allfound.erase(std::remove_if(allfound.begin(), allfound.end(), [&dels](const char &c) {
                            return dels.find(c) != std::string::npos;
                        }),
                        allfound.end());

        
        
        std::stringstream ss(allfound);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> vstrings(begin, end);
        std::copy(vstrings.begin(), vstrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
        //allfound = std::regex_replace(allfound, std::regex("|||"), " ");


        for(int j = 0; j <vstrings.size(); j++){
            size_t found = vstrings[j].find("|||");
            if (found != string::npos){
                //cout<<"Found:  "<<vstrings[j]<<" \n";
                vstrings.push_back(vstrings[j].substr(found, vstrings[j].length() - 1));
                vstrings[j] = vstrings[j].substr(0, found - 1);
            }
            //cout<<vstrings[j]<<endl;
            if (std::find(cat_word.begin(), cat_word.end(), vstrings[j]) == cat_word.end()) {
                // someName not in name, add it
                cat_word.push_back(vstrings[j]);
            }
            std::map<char, int>::iterator it = dic.find(type);
            if (it != dic.end()){
                it->second.push_back(lookup(vstrings[j]));
            }
            else{
                dic.insert(std::pair<string,int> (type, lookup(vstrings[j])));
            }
        }
        
        

        //cout<<allfound<<endl;
    }
}


void pr(){
    int rows=dic.size();
    int cols=cat_word.size();
    int feature[rows][cols];
    
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            if(dic[i].size()< j ) break;
            feature[i][j] =  dic[i][j];
        }
    }

    int mbti[rows];
    for (int i=0;i<rows;i++)
        mbti[i] = 1+i*i*i;


    DMatrixHandle mat_mbti[1];
    XGDMatrixCreateFromMat((float *) feature, rows, cols, -1, &mat_mbti[0]);
    XGDMatrixSetFloatInfo(mat_mbti[0], "label", mbti, rows);
    // read back the labels, just a sanity check
    bst_ulong bst_result;
    const float *out_floats;
    XGDMatrixGetFloatInfo(mat_mbti[0], "label" , &bst_result, &out_floats);
    for (unsigned int i=0;i<bst_result;i++)
        std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;

    // create the booster and load some parameters
    BoosterHandle h_booster;
    XGBoosterCreate(mat_mbti, 1, &h_booster);
    XGBoosterSetParam(h_booster, "booster", "gbtree");
    XGBoosterSetParam(h_booster, "objective", "reg:linear");
    XGBoosterSetParam(h_booster, "max_depth", "5");
    XGBoosterSetParam(h_booster, "eta", "0.1");
    XGBoosterSetParam(h_booster, "min_child_weight", "1");
    XGBoosterSetParam(h_booster, "subsample", "0.5");
    XGBoosterSetParam(h_booster, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster, "num_parallel_tree", "1");

    // perform 200 learning iterations
    for (int iter=0; iter<200; iter++)
        XGBoosterUpdateOneIter(h_booster, iter, mat_mbti[0]);

    // predict
    const int sample_rows = 5;
    float test[sample_rows][cols];
    for (int i=0;i<sample_rows;i++)
        for (int j=0;j<cols;j++)
            test[i][j] = (i+1) * (j+1);
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float *) test, sample_rows, cols, -1, &h_test);
    bst_ulong out_len;
    const float *f;
    XGBoosterPredict(h_booster, h_test, 0,0,&out_len,&f);

    for (unsigned int i=0;i<out_len;i++)
        std::cout << "prediction[" << i << "]=" << f[i] << std::endl;


    // free xgboost internal structures
    XGDMatrixFree(mat_mbti[0]);
    XGDMatrixFree(h_test);
    XGBoosterFree(h_booster);
}

int main()
{
    
    read_csv();
    pr();
    return 0;
}