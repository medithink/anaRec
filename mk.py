# fileまでの絶対pathを作成する
# text => 相対path
# return => 絶対path

def mk_path(text):
    import os
    # ファイルへの絶対pathを作成
    # text => mecab_pyからの相対pathを記載
    name = os.path.dirname(os.path.abspath(__name__))   #スクリプトのあるディレクトリの絶対パスを取得
    joined_path = os.path.join(name, text)      #絶対パスと相対パスをくっつける
    filename = os.path.normpath(joined_path)    #正規化して絶対パスにする
    return filename