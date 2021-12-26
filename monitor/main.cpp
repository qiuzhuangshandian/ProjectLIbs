

#include <QApplication>
#include "chartsView.h"
#include "helper.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    setCode();
    ChartsView w(100);
    w.show();
    return a.exec();
}
