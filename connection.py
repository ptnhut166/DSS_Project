from streamlit.connections import ExperimentalBaseConnection
import os
import pandas as pd
import streamlit as st
class KaggleDatasetConnection(ExperimentalBaseConnection):
    def _connect(self):
        # Set Kaggle credentials
        os.environ["KAGGLE_USERNAME"] = self._secrets.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = self._secrets.KAGGLE_KEY
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.conn = KaggleApi()
    def list(self, path, ttl):
        @st.cache_data(ttl=ttl)
        def _list(path=path):
            self.conn.authenticate()
            owner, dataset = path.split("/")
            file_list = self.conn.datasets_list_files(owner, dataset)
            new_file_list = []
            for file in file_list["datasetFiles"]:
                new_file_list.append(file["name"])
            return new_file_list
        return _list(path)
    def get(self, path, filename, ttl):
        @st.cache_data(ttl=ttl)
        def _get(path=path,filename=filename):
            self.conn.authenticate()
            self.conn.dataset_download_file(dataset=path,file_name=filename)
            df = pd.DataFrame()
            if filename.endswith(".csv"):
                df = pd.read_csv(filename)
            elif filename.endswith(".json"):
                df = pd.read_json(filename)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(filename)
            return df
        return _get()
