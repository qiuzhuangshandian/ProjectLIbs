#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
//    chart = new ChartsView(1000,this);

}

MainWindow::~MainWindow()
{
    delete ui;
//    delete chart;
}

