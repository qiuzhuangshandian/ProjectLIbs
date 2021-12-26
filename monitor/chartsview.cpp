#include <QDebug>
#include "ChartsView.h"
#include <QtGui/QResizeEvent>
#include <QtWidgets/QGraphicsScene>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QSplineSeries>
#include <QtWidgets/QGraphicsTextItem>
#include <QtGui/QMouseEvent>
#include <QValueAxis>
#include "helper.h"

#define X_Width     5           //宽度5S

ChartsView::ChartsView(int YRange, QWidget *parent)
    : QGraphicsView(new QGraphicsScene, parent),
      m_coordX(0),
      m_coordY(0),
      m_chart(0)
{

    setDragMode(QGraphicsView::NoDrag);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    m_chart = new QChart;

    initLineSeries( YRange);


    m_chart->setAcceptHoverEvents(true);

    setRenderHint(QPainter::Antialiasing);
    scene()->addItem(m_chart);


    m_coordX = new QGraphicsSimpleTextItem(m_chart);
    m_coordX->setPos(m_chart->size().width()/2 + 100, m_chart->size().height());
    m_coordX->setText("");
    m_coordX->setPen(QColor(20,164,226));
    m_coordY = new QGraphicsSimpleTextItem(m_chart);
    m_coordY->setPos(m_chart->size().width()/2 - 50, m_chart->size().height());
    m_coordY->setText("");
    m_coordY->setPen(QColor(226,20,216));


    this->setMouseTracking(true);


    MSecBase = QDateTime::currentDateTime().toMSecsSinceEpoch();
}

void ChartsView::initLineSeries(int YRange)
{
    QColor colors[3] = {QColor(20,164,226),QColor(36,20,226),QColor(226,20,216)};

    //创建X轴和Y轴
    axisX = new QValueAxis;
    axisX->setLabelFormat("%ds");
    axisX->setTickCount(X_Width);     //定义X有多少个表格
    axisX->setRange(0,X_Width);

    axisY = new QValueAxis;
    axisY->setRange(-YRange,YRange);
    axisY->setTitleText("纹波值");
    axisY->setTickCount(9);     //定义Y有多少个表格

    for(int i =0; i<3;i++)
    {
        series[i].setColor(colors[i]);
        series[i].setName(QString("测值%1").arg(i));

        series[i].setVisible(true);

        m_chart->addSeries(&series[i]);
        m_chart->setAxisX(axisX,&series[i]);
        m_chart->setAxisY(axisY,&series[i]);

        m_Line[i] = new QGraphicsSimpleTextItem(m_chart);
        m_Line[i]->setPos(34+i*40, m_chart->size().height()- 20);
        m_Line[i]->setText(QString("测值%1").arg(i));
        m_Line[i]->setPen(colors[i]);
    }

    //设置底部
    m_chart->legend()->setVisible(false);

}
void ChartsView::initMSecBase(int YRange)           //初始化MSecBase,以MSecBase为开头
{
    axisX->setMin(0);
    axisX->setMax(X_Width);

    axisY->setMin(-YRange);
    axisY->setMax(YRange);

    for(int i=0;i<3;i++)
    {
        series[i].clear();
        series[i].clear();
        series[i].append(QPointF(0,0));
    }

    MSecBase = QDateTime::currentDateTime().toMSecsSinceEpoch();
}

void ChartsView::clearMSecBase()
{
    MSecBase = QDateTime::currentDateTime().toMSecsSinceEpoch();
}

void ChartsView::setShowLine(int flag)           //0~2:显示某个曲线  否则的话显示所有曲线
{
    for(int i=0;i<3;i++)
    {
        if(flag>=0&&flag<=2)
        {
            if(flag==i)
                 series[i].show();
            else
                 series[i].hide();
        }
        else
            series[i].show();
    }

}

void ChartsView::addLineSeries(int i,float value)
{
   float currentS =(QDateTime::currentDateTime().toMSecsSinceEpoch() - MSecBase)/1000.0;   //1s=1000MS
   if(i>=3)    return;

   if(currentS>=axisX->max())
   {
        axisX->setMin(currentS-X_Width);
        axisX->setMax(currentS);

   }
   if(value>=qMax(qAbs(axisY->max()),qAbs(axisY->min())))
   {
        axisY->setMin(-value*1.1);
        axisY->setMax(value*1.1);
   }

   //定时清除以前不要的曲线数据
   if(series[i].count()>600)
   {
       series[i].removePoints(0,series[i].count()/2);
   }
   series[i].append(QPointF(currentS,value));
}


void ChartsView::addLineSeries(int i,float s,float value)
{
    if(s>=axisX->max())
    {
         axisX->setMin(s-X_Width);
         axisX->setMax(s);
    }
    if(value>=qMax(qAbs(axisY->max()),qAbs(axisY->min())))
    {
         axisY->setMin(-value*1.1);
         axisY->setMax(value*1.1);
    }

    series[i].append(QPointF(s,value)) ;
}


void ChartsView::resizeEvent(QResizeEvent *event)
{
    if (scene()) {
        scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
         m_chart->resize(event->size());
         m_coordX->setPos(m_chart->size().width()/2 + 100, m_chart->size().height() - 20);
         m_coordY->setPos(m_chart->size().width()/2 - 50, m_chart->size().height() - 20);

         for(int i=0;i<3;i++)
         {
             m_Line[i]->setPos(34+i*40, m_chart->size().height()- 20);
             m_Line[i]->setText(QString("测值%1").arg(i));
         }
    }
    QGraphicsView::resizeEvent(event);
}

void ChartsView::mouseMoveEvent(QMouseEvent *event)
{
    m_coordX->setText(QString("值:%1mV").arg(QString::asprintf("%.2f",m_chart->mapToValue(event->pos()).y())));
    m_coordY->setText(QString("时间:%1S").arg(QString::asprintf("%.2f",m_chart->mapToValue(event->pos()).x())));


    emit mouseMoveisChange(QPointF(m_chart->mapToValue(event->pos()).x(),m_chart->mapToValue(event->pos()).y()));

    QGraphicsView::mouseMoveEvent(event);
}
