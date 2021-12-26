#ifndef VIEW_H
#define VIEW_H
#include <QtWidgets/QGraphicsView>
#include <QtCharts/QChartGlobal>
#include <QLineSeries>
#include <QValueAxis>
#include <QDateTime>
#include <QSplineSeries>
QT_BEGIN_NAMESPACE
class QGraphicsScene;
class QMouseEvent;
class QResizeEvent;
QT_END_NAMESPACE

QT_CHARTS_BEGIN_NAMESPACE
class QChart;
QT_CHARTS_END_NAMESPACE
class Callout;
QT_CHARTS_USE_NAMESPACE

class ChartsView: public QGraphicsView
{
    Q_OBJECT
    qint64    MSecBase;


public:
    ChartsView(int YRange, QWidget *parent = 0);
    QLineSeries series[3];
    void initLineSeries(int YRange);
    void initMSecBase(int YRange);            //初始化MSecBase,以MSecBase为开头
    void clearMSecBase();                    //清0,从头开始
    void addLineSeries(int i,float value);              //添加value值到第几个线上面
    void addLineSeries(int i,float s,float value);       //添加s时间和value值到第几个线上面
    void setShowLine(int flag);           //0~2:显示某个曲线  否则的话显示所有曲线


protected:
    void resizeEvent(QResizeEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

public slots:

private:
    QGraphicsSimpleTextItem *m_Line[3];
    QGraphicsSimpleTextItem *m_coordX;
    QGraphicsSimpleTextItem *m_coordY;
    QChart *m_chart;
    QValueAxis *axisX;
    QValueAxis *axisY;

signals:
     void mouseMoveisChange(QPointF point);
};

#endif
